"""The whole pipeline: identify -> read colour -> plan -> generate. Returns JSON-ready data (notebook cell 8)."""
import time
from PIL import Image
from .colors import PALETTE, dominant_color, nearest_color_name, pil_to_b64
from .classifier import classify, FRIENDLY
from .planner import plan_outfit

CONF_MIN = 0.55            # below this the model says "not sure" instead of guessing
ROLES = ["auto", "top", "bottom", "dress", "shoes"]
_hex = lambda rgb: "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def recommend(image, gender="Men", style="casual", seed=None, role_override="auto", generate=True):
    """Identify the uploaded garment, then generate whichever of top / bottom / shoes is missing."""
    img = (Image.open(image) if isinstance(image, str) else image).convert("RGB")
    label, role, conf, top = classify(img)
    rgb = dominant_color(img)
    name = nearest_color_name(rgb)
    res = {"label": label, "friendly": FRIENDLY[label], "role": role, "confidence": conf,
           "top": [{"label": FRIENDLY[n], "p": p} for n, p in top],
           "color": {"name": name, "hex": _hex(rgb)}, "uncertain": False, "looks": []}
    if role_override in ROLES[1:]:
        res.update(label=None, friendly=None, role=role_override)      # the person told us what it is
    elif conf < CONF_MIN:
        res["uncertain"] = True
        res["summary"] = (f"Not sure what this is (best guess {FRIENDLY[label]}, {conf:.0%}). "
                          f"Colour {name}. Choose the garment type and try again.")
        return res
    plan = plan_outfit(res["role"], rgb, gender, style)
    res["looks"] = [{"title": l["title"], "rule": l["rule"], "items": []} for l in plan["looks"]]
    if generate:
        from .generator import generate_item
        t0 = time.time()
        for look, out in zip(plan["looks"], res["looks"]):
            for it in look["items"]:
                im, de, ok = generate_item(it["color"], it["noun"], gender, it["role"], seed=seed)
                out["items"].append({"role": it["role"], "color": it["color"], "hex": _hex(PALETTE[it["color"]]),
                                     "noun": it["noun"], "delta_e": round(de, 1), "type_ok": ok,
                                     "image": "data:image/jpeg;base64," + pil_to_b64(im, 512)})
        who = f"{res['friendly']} ({conf:.0%})" if res["label"] else f"{res['role']} (set by you)"
        res["summary"] = f"Identified {who}, colour {name}. Generated {sum(len(l['items']) for l in res['looks'])} items in {time.time() - t0:.0f}s."
    return res
