"""Colour tools: palette, Lab distance, garment cut-out, garment colour (notebook cell 2)."""
import io, base64, colorsys, time, random

import cv2

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Fashion colour names. They match the catalogue's `baseColour` values where possible.
PALETTE = {
    "Black": (20, 20, 20), "White": (245, 245, 245), "Off White": (238, 232, 215),
    "Cream": (250, 240, 200), "Beige": (222, 205, 175), "Grey": (128, 128, 128),
    "Charcoal": (54, 69, 79), "Navy Blue": (25, 35, 80), "Blue": (45, 100, 200),
    "Teal": (0, 128, 128), "Green": (40, 140, 70), "Olive": (110, 115, 40),
    "Khaki": (170, 160, 100), "Mustard": (205, 160, 30), "Yellow": (240, 220, 50),
    "Orange": (240, 130, 30), "Rust": (170, 75, 35), "Red": (200, 30, 40),
    "Maroon": (110, 20, 35), "Pink": (240, 150, 180), "Purple": (110, 50, 150),
    "Lavender": (190, 170, 225), "Brown": (100, 65, 40), "Tan": (190, 150, 100),
    "Peach": (255, 200, 170),
}
NAMES = list(PALETTE)


def rgb_to_lab(rgb):
    """sRGB (0-255) -> CIE Lab. Works on one colour or an (n,3) array."""
    c = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


PAL_LAB = rgb_to_lab(np.array([PALETTE[n] for n in NAMES]))


def delta_e(rgb_a, rgb_b):
    return float(np.linalg.norm(rgb_to_lab(rgb_a)[0] - rgb_to_lab(rgb_b)[0]))


def nearest_color_name(rgb, exclude=()):
    d = np.linalg.norm(PAL_LAB - rgb_to_lab(rgb)[0], axis=1)
    for i in np.argsort(d):
        if NAMES[i] not in exclude:
            return NAMES[i]


def _small(img, max_side=256):
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    return np.asarray(im)


def _largest_blob(mask):
    """Close small gaps, then keep only the biggest connected region (drops speckle and stray objects)."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    return lab == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _lab(a):
    """uint8 RGB array (h,w,3) -> CIE Lab float array, via OpenCV (L 0-100, a/b about -128..127)."""
    lab = cv2.cvtColor(a.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return lab


def segment_garment(a):
    """Boolean mask of the garment in an RGB uint8 array. Returns (mask, method); mask is None if nothing reliable.
    1) Background model: the border pixels are the background. Learn their colour range (shadows, folds, gradients)
       with k-means and call everything far from that range the garment. Used only when the border is consistent.
    2) GrabCut for busy backgrounds / people / hangers, seeded with 'the garment is around the middle'."""
    h, w, _ = a.shape
    lab = _lab(a)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    kb = min(4, len(np.unique(border.round(1), axis=0)))               # a perfectly flat border has 1 colour
    km = KMeans(n_clusters=kb, n_init=3, random_state=0).fit(border)
    d_border = np.linalg.norm(border[:, None, :] - km.cluster_centers_[None], axis=2).min(axis=1)
    if np.percentile(d_border, 90) < 14:                                  # border is a consistent background
        d_all = np.linalg.norm(lab.reshape(-1, 1, 3) - km.cluster_centers_[None], axis=2).min(axis=1).reshape(h, w)
        m = _largest_blob(d_all > 22)
        if 0.02 < m.mean() < 0.92:
            return m, "background model"
    try:
        gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        my, mx = max(1, int(h * .06)), max(1, int(w * .06))
        gc[my:h - my, mx:w - mx] = cv2.GC_PR_FGD                          # probably garment
        gc[int(h * .35):int(h * .65), int(w * .35):int(w * .65)] = cv2.GC_FGD    # surely garment (middle)
        gc[:my], gc[-my:], gc[:, :mx], gc[:, -mx:] = (cv2.GC_BGD,) * 4    # surely background (frame)
        cv2.grabCut(np.ascontiguousarray(a[:, :, ::-1]), gc, None,
                    np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
        m = _largest_blob((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
        if 0.05 < m.mean() < 0.85:
            return m, "grabcut"
    except cv2.error:
        pass
    return None, "centre crop"


def garment_pixels(img, max_side=256):
    """(RGB pixels of the garment, method used). Falls back to the middle of the picture."""
    a = _small(img, max_side)
    mask, how = segment_garment(a)
    if mask is not None:
        return a[mask].astype(np.float64), how
    h, w, _ = a.shape
    return a[int(h * .3):int(h * .7), int(w * .3):int(w * .7)].reshape(-1, 3).astype(np.float64), how


def crop_garment(img, max_side=384, margin=0.06):
    """Crop the photo to the garment so the classifier sees an item, not a whole scene."""
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    a = np.asarray(im)
    mask, _ = segment_garment(a)
    if mask is None:
        return im
    ys, xs = np.where(mask)
    h, w = mask.shape
    py, px = int((ys.max() - ys.min()) * margin) + 2, int((xs.max() - xs.min()) * margin) + 2
    return im.crop((max(0, xs.min() - px), max(0, ys.min() - py), min(w, xs.max() + px + 1), min(h, ys.max() + py + 1)))


def dominant_color(img, k=3, seed=0):
    """Main colour of the garment (largest k-means cluster of its pixels), as an RGB tuple.
    If the segmentation was only approximate (GrabCut / centre crop) and the biggest cluster is a dull white/grey
    (probably leftover background), a clearly coloured cluster of similar size wins instead."""
    px, how = garment_pixels(img)
    if len(px) > 6000:
        px = px[np.random.default_rng(seed).choice(len(px), 6000, replace=False)]
    k = min(k, len(np.unique(px.astype(int), axis=0)))
    if k <= 1:
        return tuple(int(v) for v in px.mean(axis=0))
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(px)
    sizes = np.bincount(km.labels_, minlength=k)
    best = int(sizes.argmax())
    if how != "background model":
        lab = _lab(km.cluster_centers_.reshape(1, -1, 3).astype(np.uint8))[0]
        chroma = np.linalg.norm(lab[:, 1:], axis=1)
        if chroma[best] < 12 and lab[best, 0] > 60:
            lively = [i for i in range(k) if chroma[i] > 30 and sizes[i] >= 0.5 * sizes[best]]
            if lively:
                best = max(lively, key=lambda i: sizes[i])
    return tuple(int(v) for v in km.cluster_centers_[best])


def pil_to_b64(img, size=220):
    im = img.convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()