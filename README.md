# 🧠 Structure-Aware Adversarial Attacks – Experimental Research

This repository contains the full implementation, results, and thesis of my research on **structure-aware adversarial image generation**.

> 📜 The full report, with methodology and analysis, is available in [`adversarial_images_thesis_Tidiane_Ciavarella.pdf`](./adversarial_images_thesis_Tidiane_Ciavarella.pdf).

---

## 📌 Project Overview

This is not a library, nor a tutorial.
This is **research-grade code** — designed for experiments, automation, and insight.

🧪 The project explores **4 distinct adversarial attack strategies** based on structural principles:

1. **Edge-guided attacks** — using Sobel filters to constrain perturbations
2. **Mask-based universal noise** — optimized neutral tensors applied to all images
3. **YCbCr chrominance attacks** — modifying only chrominance channels via mutation
4. **BestPixels** — deterministic pixel targeting based on model impact

---

## 📁 Project Structure

```text
📂 1_base/               ← Shared evolutionary components
 ├── ea_base.py              # Evolutionary algorithm base class
 ├── individual_test_*.py    # Controlled environment tests
 ├── show_base.py            # Visualization of image evolution
 └── testing_zone_*.py       # Automated experiments (HPC-ready)

📂 2_edges/             ← Edge-guided attack implementation  
📂 3_noise/             ← Universal noise via masks  
📂 4_best_pixels/       ← BestPixels: deterministic selection  
📂 1.5_YCbCr/           ← Chrominance-based perturbation logic

📂 dog_images/          ← Dataset used for testing  
📄 tools.py             ← Shared tools: mutation, batching, model handling  
📄 LICENSE  
📄 adversarial_images_thesis_Tidiane_Ciavarella.pdf
```

Each strategy folder contains:

* `ea.py`: the evolutionary algorithms implementations
* `individual_test.py`: controlled tests in constrained environments
* `show.py`: reconstruction and visualization of adversarial results
* `testing_zone.py`: designed for automated HPC experimentation

---

## ⚙️ Requirements

* Python ≥ 3.8
* PyTorch
* timm
* Hugging Face `transformers`
* NumPy, Pandas, Matplotlib, Seaborn
* Pillow, scikit-image

Install via:

```bash
pip install -r requirements.txt
```

---

## 📦 Main Imports

```python
from transformers import (
    AutoModelForImageClassification,
    AutoProcessor,
    AutoImageProcessor
)
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import requests
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import sys
import math
import itertools
import time
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import timm
```

---

## 🧠 Purpose

This repository serves as a **proof of concept** to support a full research thesis on adversarial robustness.
It focuses on **interpretable, structured perturbations** rather than black-box or random noise.

It’s intended for:

* Researchers exploring robustness and structure in adversarial ML
* Developers interested in evolution-based optimization
* Anyone seeking **non-random**, **interpretable**, and **efficient** adversarial generation

---

## 📬 Contact

Want to discuss, challenge, or build on this work?
Feel free to reach out:

📧 \[[tidarell.pro@gmail.com](mailto:tidarell.pro@gmail.com)]

---

> © Tidiane Ciavarella — All methods and insights are original unless stated otherwise.
> If you use this work, please cite or reference the original repository.
