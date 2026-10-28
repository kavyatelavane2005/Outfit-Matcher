# Outfit Matcher: Deep Learning Garment Identification & Outfit Synthesis
## Comprehensive Project Documentation & Technical Report

---

### Executive Summary

**Outfit Matcher** is a hybrid Deep Learning and Computer Vision application designed to solve personal wardrobe styling challenges. 

A user uploads a photo of an individual clothing item (such as a shirt, trousers, dress, or shoes). The system:
1. Isolates the garment from backgrounds, shadows, and surroundings using computer vision techniques.
2. Identifies the clothing category using a fine-tuned Convolutional Neural Network (**ResNet18**).
3. Extracts the primary garment color using **K-Means clustering** in CIE Lab space.
4. Generates three stylistically coordinated looks using formal **Color Theory** rules.
5. Synthesizes photorealistic images of missing complementary pieces using **SD-Turbo (Latent Diffusion Model)**, complete with automated quality validation.

---

## 1. High-Level Architecture & Workflow

```text
 Uploaded Image
       │
       ▼
┌──────────────────────────────────────────────┐
│ 1. Preprocessing & Garment Segmentation      │
│    (K-Means Background Modeler / GrabCut)    │
└──────────────────────────────────────────────┘
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ 2. Garment Classification    │  │ 3. Dominant Color Extraction │
│    - ResNet18 (Fine-Tuned)   │  │    - CIE Lab Conversion      │
│    - 15 Fashion Classes      │  │    - K-Means Clustering      │
│    - Top / Bottom / Dress /  │  │    - Nearest Named Palette   │
│      Shoes Role Mapping      │  │      Match (25 Colors)       │
└──────────────────────────────┘  └──────────────────────────────┘
       │                                        │
       └───────────────────┬────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Wardrobe & Outfit Planner                           │
│    - Evaluates Base Color (Neutral vs. Vivid)          │
│    - Generates 3 Curated Looks:                        │
│        * Look 1: Classic Neutral Pairing               │
│        * Look 2: Tone-on-tone (Analogous Hue)          │
│        * Look 3: Contrast (Complementary Hue)          │
│    - Determines Missing Roles & Fashion Nouns          │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Generative Image Synthesis & Quality Filter         │
│    - Stability AI SD-Turbo (Text-to-Image Diffusion)   │
│    - In-loop Automated Evaluation:                     │
│        * ΔE Color Match Accuracy                       │
│        * ResNet18 Garment Type Verification            │
│    - Retries and selects top candidate                 │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 6. Output Delivery & Web Interface                     │
│    - Flask API (JSON response with Base64 previews)    │
│    - Interactive HTML5 / CSS3 / JavaScript Webpage     │
└────────────────────────────────────────────────────────┘
```

---

## 2. Deep Dive: Core Components & Modules

### 2.1 Garment Segmentation & Color Extraction (`outfit_matcher/colors.py`)
- **Background Filtering**:
  - Checks perimeter pixels across the image border. If border pixels are uniform, a background model isolates the object via K-Means clustering.
  - For complex scenes, it initiates OpenCV's **GrabCut** algorithm with a center-focused rectangular prior.
  - Computes the bounding box of the largest connected mask component and crops the garment with safe padding.
- **Color Extraction**:
  - Converts RGB pixels to **CIE Lab** color space (which aligns with human perceptual differences).
  - Uses **K-Means clustering** ($k=3$) over garment pixels to locate the dominant cluster centroid.
  - Evaluates perceptual color difference using the **Euclidean Distance ($\Delta E$)** against a curated fashion palette of 25 standard colors (e.g., Navy Blue, Olive, Khaki, Rust, Cream, Charcoal).

---

### 2.2 Deep Transfer Learning Classifier (`outfit_matcher/classifier.py`)
- **Base Backbone**: Pretrained **ResNet18** (ImageNet weights).
- **Architecture Customization**:
  - Replaces final fully-connected linear layer (`model.fc`) to classify into 15 specific fashion classes:
    - **Top**: T-shirts, Shirts, Tops, Sweatshirts, Jackets.
    - **Bottom**: Jeans, Trousers, Shorts, Skirts.
    - **Dress**: Dresses.
    - **Shoes**: Casual Shoes, Formal Shoes, Sports Shoes, Sandals, Heels.
- **Two-Phase Training Protocol (`train.py`)**:
  - **Phase 1 (Head Only)**: Freezes convolutional feature extractor; trains only the new FC layer for 3 epochs with Adam ($\text{lr} = 10^{-3}$).
  - **Phase 2 (Fine-Tuning)**: Unfreezes `layer4` (the final residual block) and trains with Adam ($\text{lr} = 10^{-4}$) for 4 epochs with batch-norm statistics frozen.
- **Inference Preprocessing**:
  - Converts non-standard aspect ratios into a canonical 3:4 portrait format by padding with median background tone (avoiding image distortion or warping).
  - Resizes to $128 \times 96$ and standardizes with ImageNet mean and variance.

---

### 2.3 Outfit Planner & Fashion Color Theory (`outfit_matcher/planner.py`)
Outfits follow classical fashion styling formulas:
- **For Neutral Inputs** (White, Off White, Cream, Beige, Grey, Charcoal, Navy Blue, Brown, Tan, Khaki):
  - Uses deterministic styling pairings (e.g., Cream pairs with Navy Blue, Brown, and Olive).
- **For Chromatic/Color Inputs**:
  - Uses the HSL color wheel to determine:
    1. **Look 1 (Safe Neutral)**: Balances vivid hues against grounding tones (Beige, Navy Blue, Grey).
    2. **Look 2 (Analogous Hue)**: Picks adjacent color wheel coordinates ($+28.8^\circ$), muted and darkened for tone-on-tone elegance.
    3. **Look 3 (Complementary Contrast)**: Picks opposite color wheel coordinates ($+180^\circ$) in controlled saturation.
- **Rule Resolution**:
  - If a user uploads a **Top** $\rightarrow$ Plans **Bottom** and **Shoes**.
  - If a user uploads a **Bottom** $\rightarrow$ Plans **Top** and **Shoes**.
  - If a user uploads a **Dress** $\rightarrow$ Plans **Shoes**.
  - If a user uploads **Shoes** $\rightarrow$ Plans harmonious **Bottom** and **Top**.

---

### 2.4 Generative Diffusion & Dual-Check Loop (`outfit_matcher/generator.py`)
- **Diffusion Engine**: **Stability AI SD-Turbo** (`AutoPipelineForText2Image`).
- **Low-Latency Generation**: Uses Single-step/Few-step Adversarial Diffusion Distillation (ADD) running at 2 inference steps, guidance scale $0.0$, generating $512 \times 512$ resolution outputs.
- **Automated Validation Loop**:
  For each generated item, the system executes up to 4 candidate iterations:
  1. Computes $\Delta E$ color distance between generated image dominant color and planned color target.
  2. Runs the generated image through our fine-tuned ResNet18 to confirm whether the model produced the intended clothing type.
  3. Computes a penalty score: $\text{Loss} = \Delta E + (0\text{ if type matches else } 25)$.
  4. Keeps the highest scoring attempt that satisfies $\Delta E \le 25$ and garment type validation.

---

### 2.5 Web Server & Application Interface (`app.py`, `website/`)
- **Backend**: Flask REST API.
  - `GET /` $\rightarrow$ Delivers responsive frontend UI.
  - `GET /api/health` $\rightarrow$ Checks hardware availability (GPU/CPU) and model checkpoint readiness.
  - `POST /api/match` $\rightarrow$ Accepts multipart form payload (`photo`, `gender`, `style`, `role`). Guarded by a thread-level GPU lock to serialize inference requests.
- **Frontend**: Plain HTML5, CSS3, and JavaScript:
  - Drag-and-drop file upload with live preview.
  - Gender selector (Men/Women) and Style selector (Casual/Formal).
  - Role override dropdown (Auto / Top / Bottom / Dress / Shoes).
  - Responsive cards displaying the detected garment, confidence metric, dominant color hex badge, and the 3 generated looks with synthesized images.

---

## 3. Project Directory Structure

```text
outfit-matcher/
├── app.py                         # Flask web server and API endpoints
├── train.py                       # ResNet18 transfer learning training script
├── requirements.txt               # Project library dependencies
├── README.md                      # Quick-start guide
├── PROJECT_DOCUMENTATION.md       # Comprehensive system report
│
├── outfit_matcher/                # Core Python package
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Path definitions, device detection (CUDA/CPU)
│   ├── colors.py                  # CV background removal, Lab color conversion, K-Means
│   ├── classifier.py              # ResNet18 definition, portrait padding, classifier logic
│   ├── planner.py                 # Color theory rules, palette mappings, outfit matrices
│   ├── generator.py               # SD-Turbo generation, prompt builder, verification loop
│   └── pipeline.py                # Pipeline coordinator (classify -> plan -> generate)
│
├── models/
│   ├── .gitkeep
│   └── garment_classifier.pt      # Fine-tuned ResNet18 model weights (~43 MB)
│
├── website/
│   └── index.html                 # Complete Single Page Application UI (HTML/CSS/JS)
│
├── docs/                          # Detailed markdown docs
│   ├── CONCEPTS.md                # Theoretical background
│   ├── HOW_IT_WORKS.md            # Pipeline specifications
│   └── SETUP.md                   # Environment setup instructions
│
└── notebook/
    └── outfit_matcher_lite_colab.ipynb  # Original Google Colab research notebook
```

---

## 4. How to Run the Project (Daily Operations)

Since both the classifier model (`models/garment_classifier.pt`) and the Hugging Face SD-Turbo weights are saved and cached locally, no further downloads are needed.

### Running Locally:
1. Open PowerShell or Command Prompt.
2. Navigate to your project directory:
   ```powershell
   cd c:\Users\kavya\OneDrive\Desktop\Project\DL\outfit-matcher
   ```
3. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate
   ```
4. Start the server:
   ```powershell
   python app.py
   ```
5. Open your web browser and navigate to:
   ```text
   http://localhost:5000
   ```

---

## 5. Technology Stack Summary

| Domain | Technology / Library | Role |
|---|---|---|
| **Deep Learning Framework** | PyTorch (`torch`, `torchvision`) | Neural network training and evaluation |
| **Generative AI** | Hugging Face `diffusers`, `transformers`, `accelerate` | SD-Turbo latent diffusion generation |
| **Computer Vision** | OpenCV (`opencv-python-headless`), Pillow (`PIL`) | Image segmentation, GrabCut, morphological filtering |
| **Scientific Computing** | NumPy, Scikit-learn, Pandas | K-Means clustering, dataset splitting, metrics |
| **Backend Web Server** | Flask | REST API and static asset hosting |
| **Frontend UI** | HTML5, Modern CSS, Vanilla JavaScript | Client-side presentation and interaction |
