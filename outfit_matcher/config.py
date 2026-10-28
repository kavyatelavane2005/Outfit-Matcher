import os, random
import numpy as np, torch

device = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "garment_classifier.pt")


def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
