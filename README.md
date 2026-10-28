# Outfit Matcher

Upload a photo of **one** clothing item (shirt, pants, shoes or a dress). The system identifies it, reads its
colour, plans three matching looks with colour-theory rules, and **generates images** of the missing pieces.

```
photo -> [crop to garment] -> [fine-tuned ResNet18] -> "shirt" -> colour rules -> [SD-Turbo] -> missing pieces
                                (we train this)                    (3 looks)     (pretrained)
```

## Quick start
```bash
pip install -r requirements.txt
python train.py        # once: trains the classifier, saves models/garment_classifier.pt (~3 min on a GPU)
python app.py          # starts the website
```
Open **http://localhost:5000**, upload a photo, press **Match my outfit**.
A GPU is strongly recommended (about 1 minute per photo on a T4; much slower on CPU).
Full instructions, Colab option and troubleshooting: [docs/SETUP.md](docs/SETUP.md).

## What is in this folder
| Path | Purpose |
|---|---|
| `website/index.html` | The website (upload page, results, project explanation). Plain HTML, CSS and JavaScript |
| `app.py` | Flask server: serves the website and the `POST /api/match` endpoint |
| `train.py` | Fine-tunes ResNet18 on 15 garment types and saves the model |
| `outfit_matcher/colors.py` | Palette, Lab colour distance, garment cut-out, dominant colour |
| `outfit_matcher/planner.py` | Colour-theory rules that choose the three looks |
| `outfit_matcher/classifier.py` | Model definition, image preparation, `classify()` |
| `outfit_matcher/generator.py` | SD-Turbo generation plus colour and type checks |
| `outfit_matcher/pipeline.py` | `recommend()`: identify, plan, generate, return JSON |
| `notebook/` | The original Colab notebook this project came from |
| `docs/` | Documentation: [concepts](docs/CONCEPTS.md), [how it works](docs/HOW_IT_WORKS.md), [setup](docs/SETUP.md) |

## Documentation
1. [docs/CONCEPTS.md](docs/CONCEPTS.md): the ideas behind the project (CNNs, transfer learning, diffusion, colour theory).
2. [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): the pipeline step by step, the code map, the API and design decisions.
3. [docs/SETUP.md](docs/SETUP.md): installation, running locally or on Colab, configuration, troubleshooting.

## Limits
Pairings come from rules, not from a model trained on real outfits. The classifier learned from small catalogue photos,
so one item on a plain background works best. Generated pieces are AI images, not real products. SD-Turbo has
Stability AI's licence terms: check them before commercial use.
