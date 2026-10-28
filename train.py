"""Train the garment classifier (transfer learning on ResNet18). Run once:  python train.py
Needs internet the first time (downloads the dataset and ImageNet weights). About 3 min on a T4 GPU."""
import argparse, copy
import numpy as np, torch, torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from outfit_matcher.config import device, MODEL_PATH, set_seed
from outfit_matcher.classifier import CLASS_ROLE, to_portrait, train_tf, eval_tf, build_model


class GarmentSet(torch.utils.data.Dataset):
    def __init__(self, images, labels, idx, tf):
        self.images, self.labels, self.idx, self.tf = images, labels, idx, tf
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(to_portrait(self.images[j])), int(self.labels[j])


def main(per_class, workers):
    set_seed(0)
    full = load_dataset("ashraq/fashion-product-images-small", split="train")
    meta = full.remove_columns("image").to_pandas()
    meta = meta[meta["articleType"].isin(CLASS_ROLE)]
    classes = [c for c in CLASS_ROLE if c in set(meta["articleType"])]
    sel = meta.sample(frac=1, random_state=0).groupby("articleType").head(per_class)     # cap per class
    images = [im.convert("RGB") for im in full.select(sel.index.to_list())["image"]]
    labels = np.array([classes.index(a) for a in sel["articleType"]])

    # 70% train, 15% validation (choose the best epoch), 15% test (used once)
    tr, tmp = train_test_split(np.arange(len(labels)), test_size=0.30, stratify=labels, random_state=0)
    va, te = train_test_split(tmp, test_size=0.50, stratify=labels[tmp], random_state=0)
    dl = lambda idx, tf, bs, sh=False: torch.utils.data.DataLoader(GarmentSet(images, labels, idx, tf), batch_size=bs,
                                                                   shuffle=sh, num_workers=workers)
    train_dl, val_dl, test_dl = dl(tr, train_tf, 64, True), dl(va, eval_tf, 128), dl(te, eval_tf, 128)
    print(f"{len(classes)} classes | train {len(tr)} | validation {len(va)} | test {len(te)}")

    model = build_model(len(classes)).to(device)
    counts = np.bincount(labels[tr], minlength=len(classes))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device))
    best = {"acc": 0.0, "state": None}

    def evaluate(loader):
        model.eval(); preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                preds += model(x.to(device)).argmax(1).cpu().tolist(); targets += y.tolist()
        return float(np.mean(np.array(preds) == np.array(targets))), preds, targets

    def run_epochs(epochs, params, lr, phase):
        opt = torch.optim.Adam(params, lr=lr)
        for ep in range(epochs):
            model.train()
            for m in model.modules():                       # keep pretrained batch-norm statistics fixed
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for x, y in train_dl:
                opt.zero_grad(); loss_fn(model(x.to(device)), y.to(device)).backward(); opt.step()
            acc = evaluate(val_dl)[0]
            if acc > best["acc"]:
                best.update(acc=acc, state=copy.deepcopy(model.state_dict()))
            print(f"[{phase}] epoch {ep + 1}/{epochs}  validation accuracy {acc:.1%}")

    for p in model.parameters():                            # Phase 1: train only the new final layer
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    run_epochs(3, model.fc.parameters(), 1e-3, "head only")
    for p in model.layer4.parameters():                     # Phase 2: also fine-tune the last block, gently
        p.requires_grad = True
    run_epochs(4, list(model.layer4.parameters()) + list(model.fc.parameters()), 1e-4, "fine-tune")

    model.load_state_dict(best["state"])                    # back to the best epoch
    acc, preds, targets = evaluate(test_dl)
    roles = np.array([CLASS_ROLE[c] for c in classes])
    print(f"\nTest accuracy (15 types): {acc:.1%} | role accuracy: {np.mean(roles[preds] == roles[targets]):.1%}")
    print(classification_report(targets, preds, target_names=classes, digits=2))
    torch.save({"state_dict": model.state_dict(), "classes": classes}, MODEL_PATH)
    print("Saved", MODEL_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=800)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    main(a.per_class, a.workers)
