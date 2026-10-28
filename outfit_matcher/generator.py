"""Image generation with SD-Turbo plus a colour check and a garment-type check (notebook cell 7)."""
import random, torch
from .colors import PALETTE, delta_e, dominant_color
from .classifier import classify
from .config import device

COLOR_TOLERANCE = 25      # max Lab delta-E between planned and generated colour
TYPE_PENALTY = 25         # added to the error when our classifier says it is the wrong kind of garment
_pipe = None


def get_pipe():
    """Load SD-Turbo once (first call downloads about 2-3 GB)."""
    global _pipe
    if _pipe is None:
        from diffusers import AutoPipelineForText2Image
        dt = torch.float16 if device == "cuda" else torch.float32
        try:
            p = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=dt,
                                                          variant="fp16" if device == "cuda" else None)
        except Exception:
            p = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=dt)
        _pipe = p.to(device)
        _pipe.set_progress_bar_config(disable=True)
    return _pipe


def make_prompt(color, noun, gender):
    who = "men's" if gender == "Men" else "women's"
    return (f"studio product photo of {who} {color.lower()} {noun}, isolated on a plain light grey background, "
            f"e-commerce catalogue photo, centered, sharp, high detail")


def generate_item(color, noun, gender, role, tries=4, seed=None):
    """Generate `color noun`. Each try is checked for colour and for garment type; the best try is kept.
    Returns (image, colour error, type ok)."""
    pipe = get_pipe()
    seed = random.randrange(10_000) if seed is None else seed
    target, best = PALETTE[color], None
    for t in range(tries):
        g = torch.Generator(device=device).manual_seed(seed + t)
        img = pipe(prompt=make_prompt(color, noun, gender), num_inference_steps=2,
                   guidance_scale=0.0, height=512, width=512, generator=g).images[0]
        de = delta_e(dominant_color(img), target)
        type_ok = classify(img)[1] == role
        score = de + (0 if type_ok else TYPE_PENALTY)
        if best is None or score < best[0]:
            best = (score, img, de, type_ok)
        if de <= COLOR_TOLERANCE and type_ok:
            break
    return best[1], best[2], best[3]
