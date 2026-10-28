"""Garment classifier: a ResNet18 fine-tuned on catalogue photos (notebook cells 5-6)."""
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T
from .colors import crop_garment
from .config import device, MODEL_PATH

CLASS_ROLE = {
    "Tshirts": "top", "Shirts": "top", "Tops": "top", "Sweatshirts": "top", "Jackets": "top",
    "Jeans": "bottom", "Trousers": "bottom", "Shorts": "bottom", "Skirts": "bottom",
    "Dresses": "dress",
    "Casual Shoes": "shoes", "Formal Shoes": "shoes", "Sports Shoes": "shoes", "Sandals": "shoes", "Heels": "shoes",
}
FRIENDLY = {"Tshirts": "T-shirt", "Shirts": "shirt", "Tops": "top", "Sweatshirts": "sweatshirt", "Jackets": "jacket",
            "Jeans": "jeans", "Trousers": "trousers", "Shorts": "shorts", "Skirts": "skirt", "Dresses": "dress",
            "Casual Shoes": "casual shoes", "Formal Shoes": "formal shoes", "Sports Shoes": "sports shoes",
            "Sandals": "sandals", "Heels": "heels"}

IMG_H, IMG_W = 128, 96                  # network input (portrait 3:4)
CATALOGUE_SIZE = (60, 80)               # (width, height) of the real training photos


def to_portrait(img):
    """Pad (never stretch) to 3:4 portrait using the photo's own background colour."""
    img = img.convert("RGB")
    w, h = img.size
    if abs(w / h - 0.75) < 0.05:
        return img
    a = np.asarray(img)
    bg = tuple(int(v) for v in np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0))
    nw, nh = (w, int(round(w / 0.75))) if w / h > 0.75 else (int(round(h * 0.75)), h)
    canvas = Image.new("RGB", (nw, nh), bg)
    canvas.paste(img, ((nw - w) // 2, (nh - h) // 2))
    return canvas


def prepare_for_model(img):
    """Make a real-world photo look like the training photos: crop to garment, pad to portrait, shrink."""
    return to_portrait(crop_garment(img)).resize(CATALOGUE_SIZE, Image.LANCZOS)


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])           # ImageNet statistics
train_tf = T.Compose([T.RandomResizedCrop((IMG_H, IMG_W), scale=(0.75, 1.0), ratio=(0.7, 0.85)),
                      T.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.8, 1.1), fill=(240, 240, 240)),
                      T.RandomApply([T.GaussianBlur(3, (0.1, 1.2))], p=0.3),
                      T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.2, 0.4, 0.05), T.ToTensor(), norm])
eval_tf = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor(), norm])


def build_model(n_classes, pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)         # only this layer is brand new
    return m


_state = {}


def load_model(path=MODEL_PATH):
    ckpt = torch.load(path, map_location=device)
    m = build_model(len(ckpt["classes"]), pretrained=False)
    m.load_state_dict(ckpt["state_dict"])
    _state.update(model=m.to(device).eval(), classes=ckpt["classes"])


@torch.no_grad()
def classify(img, top_k=3):
    """Garment type of any photo. Returns (label, role, confidence, top_k list)."""
    if "model" not in _state:
        load_model()
    x = eval_tf(prepare_for_model(img)).unsqueeze(0).to(device)
    p = torch.softmax(_state["model"](x), dim=1)[0].cpu().numpy()
    classes = _state["classes"]
    top = [(classes[i], float(p[i])) for i in np.argsort(-p)[:top_k]]
    return top[0][0], CLASS_ROLE[top[0][0]], top[0][1], top
