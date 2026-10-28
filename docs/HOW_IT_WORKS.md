# How Outfit Matcher works

## Architecture
```
Browser (website/index.html)
   |  POST /api/match  (photo, gender, style, role)
   v
Flask server (app.py)  --- one request at a time (GPU lock)
   v
outfit_matcher/pipeline.py : recommend()
   1. classifier.classify(photo)      crop_garment -> to_portrait -> resize 60x80 -> ResNet18 -> label, role, confidence
   2. colors.dominant_color(photo)    segment_garment -> k-means -> nearest palette name
   3. planner.plan_outfit(role, rgb)  3 looks, each with the missing items (colour + noun)
   4. generator.generate_item(...)    SD-Turbo, up to 4 tries, colour + type check
   v
JSON (identification, colour, 3 looks with base64 images)  ->  rendered in the browser
```

## Request walk-through
1. **Upload.** The page sends the photo and the options. The server rejects non-images, files over 12 MB and invalid options.
2. **Identify.** The photo is cropped to the garment, padded (never stretched) to 3:4 portrait and shrunk to 60x80 px
   so it looks like the soft, tiny training photos. ResNet18 returns probabilities for 15 types. The top type maps to a
   **role**: top, bottom, dress or shoes.
3. **Decide.** If the user chose a role, it is used. Else if confidence is under 55%, the server replies
   `uncertain: true` and the page asks the user to choose. Otherwise continue.
4. **Colour.** The garment pixels are clustered; the main colour is matched to one of 25 named colours.
5. **Plan.** `plan_outfit` returns three looks. Example for a blue shirt: Look 1 beige jeans + white sneakers,
   Look 2 analogous colour, Look 3 complementary colour. Item nouns (jeans, chinos, loafers...) depend on gender and style.
6. **Generate.** For each missing item a prompt is built and SD-Turbo draws 512x512 images (2 steps, no guidance).
   Each is checked for colour (delta-E <= 25) and role; the best of up to 4 tries is returned.
7. **Show.** The page displays what was identified, the colour, top-3 guesses and the three looks.

Which items are generated: shirt uploaded -> pants and shoes; pants -> shirt and shoes; shoes -> shirt and pants;
dress -> shoes. (Defined by `NEEDS` in `planner.py`.)

## Training (`train.py`)
- **Data:** Hugging Face dataset `ashraq/fashion-product-images-small`, 15 garment types, max 800 photos each (about 9,000 images).
- **Split:** 70% train, 15% validation, 15% test, stratified by class.
- **Model:** ImageNet-pretrained ResNet18, new final layer for 15 classes.
- **Phases:** head only (3 epochs, lr 1e-3), then fine-tune `layer4` + head (4 epochs, lr 1e-4); best validation epoch kept.
- **Output:** `models/garment_classifier.pt` containing the weights and the class list. The test report is printed.

## API
`GET /api/health` -> `{"ok": true, "device": "cuda", "model_ready": true}`

`POST /api/match` (multipart form)

| Field | Values | Default |
|---|---|---|
| `photo` | image file (JPG, PNG, WebP), max 12 MB | required |
| `gender` | `Men`, `Women` | `Men` |
| `style` | `casual`, `formal` | `casual` |
| `role` | `auto`, `top`, `bottom`, `dress`, `shoes` | `auto` |

Response (abridged):
```json
{
  "friendly": "shirt", "role": "top", "confidence": 0.93, "uncertain": false,
  "top": [{"label": "shirt", "p": 0.93}, {"label": "T-shirt", "p": 0.05}],
  "color": {"name": "Blue", "hex": "#2c64c8"},
  "looks": [{"title": "Look 1: Classic neutral pairing", "rule": "Classic neutral pairing",
             "items": [{"role": "bottom", "color": "Beige", "noun": "jeans", "hex": "#decdaf",
                        "delta_e": 9.8, "type_ok": true, "image": "data:image/jpeg;base64,..."}]}],
  "summary": "Identified shirt (93%), colour Blue. Generated 6 items in 38s."
}
```
Errors return `{"error": "..."}` with status 400 (bad input) or 503 (classifier not trained).

Try it from a terminal: `curl -F photo=@shirt.jpg -F gender=Men http://localhost:5000/api/match`

## Code map
| File | Key functions |
|---|---|
| `colors.py` | `PALETTE`, `rgb_to_lab`, `delta_e`, `nearest_color_name`, `segment_garment`, `crop_garment`, `dominant_color` |
| `planner.py` | `harmony_colors`, `shoe_color`, `ITEM_OPTIONS`, `NEEDS`, `plan_outfit` |
| `classifier.py` | `CLASS_ROLE`, `to_portrait`, `prepare_for_model`, `train_tf` / `eval_tf`, `build_model`, `load_model`, `classify` |
| `generator.py` | `get_pipe`, `make_prompt`, `generate_item`, `COLOR_TOLERANCE`, `TYPE_PENALTY` |
| `pipeline.py` | `recommend`, `CONF_MIN` |

## Design decisions (and why)
- **Prepare real photos to look like catalogue photos.** Real photos are big and sharp; training photos are tiny and
  soft. Cropping and shrinking before classifying fixed a blue shirt being read as "jacket (96%)".
- **Learn the background from the border** instead of assuming a flat colour, so shadows and folds are not counted as garment.
- **Say "not sure"** rather than plan an outfit around a wrong guess.
- **Check generated images** because text-to-image models often ignore the requested colour or draw the wrong garment.
- **Rules instead of a learned compatibility model**, because no outfit dataset is used. This is the main limitation.
- **Pad, never stretch**, photos to portrait so garment proportions are preserved.

## Tuning knobs
| Setting | File | Effect |
|---|---|---|
| `CONF_MIN` (0.55) | `pipeline.py` | Higher = asks the user more often |
| `COLOR_TOLERANCE` (25) | `generator.py` | Lower = stricter colour match, more retries |
| `tries` (4) | `generator.py` | More tries = better matches, slower |
| `PER_CLASS` / `--per-class` (800) | `train.py` | More data per class, longer training |
| `PALETTE`, `NEUTRAL_PAIRS` | `colors.py`, `planner.py` | Add colours or change stylist pairings |

## Limitations and future work
- Tiny catalogue training photos mean people wearing the item or cluttered scenes can be misclassified.
- The catalogue may contain near-duplicate product shots, so the test score can be slightly optimistic.
- A jacket counts as a "top", so no shirt is generated under it.
- **Future work:** a compatibility model trained on real outfits (e.g. Polyvore Outfits), fine-tuning on photos of worn
  items, IP-Adapter so generated pieces follow the uploaded item's pattern, and links to real products.
