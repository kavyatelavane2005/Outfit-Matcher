"""Outfit planner: colour-theory rules (notebook cell 3)."""
import colorsys
from .colors import PALETTE, nearest_color_name
# For a neutral piece we use stylist pairings. For a coloured piece we use colour harmony:
#   Look 1 = safe neutral, Look 2 = analogous (neighbouring hue, muted), Look 3 = complementary (opposite hue, muted).
NEUTRAL_PAIRS = {
    "White": ["Navy Blue", "Black", "Khaki"],       "Off White": ["Navy Blue", "Black", "Olive"],
    "Cream": ["Navy Blue", "Brown", "Olive"],       "Beige": ["Navy Blue", "Olive", "Brown"],
    "Black": ["Blue", "Grey", "Beige"],             "Grey": ["Navy Blue", "Black", "Maroon"],
    "Charcoal": ["Blue", "Beige", "Khaki"],         "Navy Blue": ["Beige", "Grey", "White"],
    "Brown": ["Beige", "Navy Blue", "Cream"],       "Tan": ["Navy Blue", "White", "Olive"],
    "Khaki": ["Navy Blue", "Black", "White"],
}
NEUTRAL_RULES = ["Classic neutral pairing", "Tone contrast", "Earthy accent"]
VIVID = {"Purple", "Pink", "Yellow", "Orange", "Lavender", "Peach", "Red"}   # too loud as a derived partner colour
EARTHY = {"Khaki", "Olive", "Beige", "Brown", "Tan", "Cream", "Rust", "Mustard"}
DARK_COOL = {"Navy Blue", "Blue", "Black", "Charcoal", "Grey", "Teal"}


def _hue_color(h, s, l, exclude):
    return nearest_color_name(tuple(255 * v for v in colorsys.hls_to_rgb(h % 1.0, l, s)), exclude=set(exclude) | VIVID)


def harmony_colors(base_rgb):
    """Three (colour name, rule) suggestions that go with base_rgb."""
    base = nearest_color_name(base_rgb)
    if base in NEUTRAL_PAIRS:
        return list(zip(NEUTRAL_PAIRS[base], NEUTRAL_RULES))
    h, l, s = colorsys.rgb_to_hls(*[v / 255 for v in base_rgb])
    used = {base}
    safe = "Beige" if (0.55 < h < 0.75) else "Navy Blue"          # blues pair better with beige than with navy
    if safe in used:
        safe = "Grey"
    used.add(safe)
    analog = _hue_color(h + 0.08, min(s, 0.45), 0.35, used)          # neighbouring hue, muted and darker
    used.add(analog)
    comp = _hue_color(h + 0.5, 0.40, 0.36, used)                     # opposite hue, muted
    if comp in used:
        comp = next(c for c in ["Black", "White", "Grey", "Khaki"] if c not in used)
    return [(safe, "Classic neutral pairing"), (analog, "Tone-on-tone (analogous)"), (comp, "Contrast (complementary)")]


def shoe_color(bottom_name, i):
    if bottom_name in EARTHY:
        order = ["White", "Brown", "Black"]
    elif bottom_name in DARK_COOL:
        order = ["White", "Black", "Brown"]
    else:
        order = ["White", "Black", "Beige"]
    choice = order[i % 3]
    return choice if choice != bottom_name else order[(i + 1) % 3]


# What to generate for each missing role: (noun used in the prompt, catalogue articleTypes to search).
ITEM_OPTIONS = {
    ("top", "Men", "casual"):       [("t-shirt", ["Tshirts"]), ("casual shirt", ["Shirts"]), ("sweatshirt", ["Sweatshirts"])],
    ("top", "Men", "formal"):       [("formal shirt", ["Shirts"])],
    ("top", "Women", "casual"):     [("top", ["Tops"]), ("t-shirt", ["Tshirts"]), ("casual shirt", ["Shirts"])],
    ("top", "Women", "formal"):     [("formal shirt", ["Shirts"]), ("blouse top", ["Tops"])],
    ("bottom", "Men", "casual"):    [("jeans", ["Jeans"]), ("chino trousers", ["Trousers"]), ("tapered trousers", ["Trousers"])],
    ("bottom", "Men", "formal"):    [("formal trousers", ["Trousers"])],
    ("bottom", "Women", "casual"):  [("jeans", ["Jeans"]), ("trousers", ["Trousers"]), ("skirt", ["Skirts"])],
    ("bottom", "Women", "formal"):  [("formal trousers", ["Trousers"]), ("pencil skirt", ["Skirts"])],
    ("shoes", "Men", "casual"):     [("sneakers", ["Casual Shoes", "Sports Shoes"]), ("canvas casual shoes", ["Casual Shoes"]), ("loafers", ["Casual Shoes"])],
    ("shoes", "Men", "formal"):     [("leather formal shoes", ["Formal Shoes"])],
    ("shoes", "Women", "casual"):   [("sneakers", ["Casual Shoes", "Sports Shoes"]), ("flats", ["Flats"]), ("sandals", ["Sandals"])],
    ("shoes", "Women", "formal"):   [("block heels", ["Heels"]), ("formal flats", ["Flats"]), ("pumps", ["Heels", "Formal Shoes"])],
}
NEEDS = {"top": ["bottom", "shoes"], "bottom": ["top", "shoes"], "dress": ["shoes"], "shoes": ["top", "bottom"]}


def plan_outfit(role, base_rgb, gender="Men", style="casual"):
    """Returns the three planned looks. Each look lists the items to generate."""
    base_name = nearest_color_name(base_rgb)
    picks = harmony_colors(base_rgb)
    looks = []
    for i, (pick, rule) in enumerate(picks):
        colors = {}
        if role == "top":
            colors["bottom"] = pick
            colors["shoes"] = shoe_color(pick, i)
        elif role == "bottom":
            colors["top"] = pick
            colors["shoes"] = shoe_color(base_name, i)
        elif role == "dress":
            colors["shoes"] = shoe_color(base_name, i) if i < 2 else "Black"
        else:  # shoes uploaded
            colors["bottom"] = pick
            partner = next(c for c, _ in harmony_colors(PALETTE[pick]) if c not in (pick, base_name))
            colors["top"] = partner
        items = []
        for need in NEEDS[role]:
            options = ITEM_OPTIONS[(need, gender, style)]
            noun, types = options[i % len(options)]
            items.append({"role": need, "color": colors[need], "noun": noun, "types": types})
        looks.append({"title": f"Look {i + 1}: {rule}", "rule": rule, "items": items})
    return {"base_name": base_name, "role": role, "looks": looks}