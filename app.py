"""Web server:  python app.py   then open http://localhost:5000
Serves the website and one API endpoint, POST /api/match."""
import io, os, threading
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from outfit_matcher.config import ROOT, MODEL_PATH, device, set_seed
from outfit_matcher.pipeline import recommend, ROLES

app = Flask(__name__, static_folder=os.path.join(ROOT, "website"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024        # 12 MB upload limit
gpu_lock = threading.Lock()                                  # one generation at a time


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(ok=True, device=device, model_ready=os.path.exists(MODEL_PATH))


@app.post("/api/match")
def match():
    if not os.path.exists(MODEL_PATH):
        return jsonify(error="Classifier not trained yet. Run `python train.py` first."), 503
    f = request.files.get("photo")
    if f is None:
        return jsonify(error="Attach a photo in the 'photo' field."), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not a readable image (use JPG, PNG or WebP)."), 400
    gender = request.form.get("gender", "Men")
    style = request.form.get("style", "casual")
    role = request.form.get("role", "auto")
    if gender not in ("Men", "Women") or style not in ("casual", "formal") or role not in ROLES:
        return jsonify(error="Invalid gender, style or role."), 400
    with gpu_lock:
        return jsonify(recommend(img, gender, style, role_override=role))


if __name__ == "__main__":
    set_seed(0)
    print(f"Device: {device}" + ("  (no GPU: generation will be slow)" if device == "cpu" else ""))
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
