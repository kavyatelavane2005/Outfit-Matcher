# Setup guide

## Requirements
- Python 3.10 or newer.
- An **NVIDIA GPU** (about 6 GB VRAM or more) is strongly recommended. A T4 generates a look in about a minute. CPU works but is very slow.
- Internet on first run: dataset (~tens of MB), ResNet18 weights, and SD-Turbo (~2-3 GB) are downloaded and cached.
- About 6 GB free disk space.

## Option A: run on your own computer
```bash
# 1. (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
#    For an NVIDIA GPU, first install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/
pip install -r requirements.txt

# 3. train the classifier (once)
python train.py                   # creates models/garment_classifier.pt

# 4. start the website
python app.py                     # open http://localhost:5000
```
Use another port with `PORT=8000 python app.py` (Windows PowerShell: `$env:PORT=8000; python app.py`).
The first photo is slower because SD-Turbo loads into memory.

## Option B: Google Colab (free T4 GPU)
Use this if you have no GPU.
1. Upload the project folder (or the zip) to Colab, or to Google Drive and mount it.
2. **Runtime > Change runtime type > T4 GPU.**
3. In notebook cells:
   ```python
   !unzip -q outfit-matcher.zip && cd outfit-matcher && pip install -q -r requirements.txt
   %cd outfit-matcher
   !python train.py
   ```
4. Start the server in the background and open it through Colab's proxy:
   ```python
   import subprocess
   subprocess.Popen(["python", "app.py"])
   from google.colab import output
   output.serve_kernel_port_as_window(5000)     # opens the website in a new tab
   ```
   Alternatives: a tunnel such as `cloudflared` or `ngrok` gives a public link (it runs on **your** GPU, so share carefully).

The original single-notebook version is in `notebook/outfit_matcher_lite_colab.ipynb`. It runs all steps and a Gradio page
with no server setup, which is the easiest way to try the model quickly.

## Using the website
1. Click the upload box (or drag a photo in). Use **one** item on a plain background.
2. Choose **Men / Women** and **Casual / Formal**.
3. Leave **Garment type** on automatic. Change it only if the model is unsure or wrong.
4. Press **Match my outfit** and wait up to a minute. Three looks appear with the missing pieces.

## Configuration
| What | How |
|---|---|
| Port | `PORT` environment variable |
| Upload limit | `MAX_CONTENT_LENGTH` in `app.py` (12 MB) |
| Training size | `python train.py --per-class 400` (faster, less accurate) |
| DataLoader workers | `python train.py --workers 0` (needed on some Windows setups) |
| Confidence threshold, colour tolerance, tries | see "Tuning knobs" in [HOW_IT_WORKS.md](HOW_IT_WORKS.md) |

## Troubleshooting
| Symptom | Fix |
|---|---|
| "Classifier not trained yet" | Run `python train.py`, then confirm `models/garment_classifier.pt` exists (no server restart needed) |
| `Device: cpu` printed at startup | PyTorch has no CUDA. Install the CUDA build of PyTorch, or use Colab |
| Out of GPU memory | Close other GPU programs; the code already uses fp16 on GPU |
| Very slow first request | SD-Turbo is downloading and loading. Later requests are faster |
| "Not sure what this is" | Choose the garment type manually, or upload a photo of just the item on a plain background |
| Wrong colour read | Make the garment fill the frame; avoid strong shadows and busy backgrounds |
| Download errors | Check your internet connection; the Hugging Face cache is in `~/.cache/huggingface` |
| Training fails on Windows with a DataLoader error | Run `python train.py --workers 0` |
| Page shows a server error | Check the terminal running `python app.py` for the traceback |

## Notes
- The server handles **one generation at a time** (a lock protects the GPU), so concurrent users wait in a queue.
- Do not expose the server publicly without adding authentication and rate limiting.
- SD-Turbo is under Stability AI's licence. Check it before commercial use.
