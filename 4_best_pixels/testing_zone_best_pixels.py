# IMPORTS

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import *

from transformers import AutoModelForImageClassification, AutoProcessor, AutoImageProcessor
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import requests
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
import math
import itertools
import time
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns



# INITIALISATION

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set model
model_name = "hilmansw/resnet18-catdog-classifier"

model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
model.eval()



# TESTING ZONE

image = Image.open("../dog_images/1.jpg")
image = transformResize(image)

tensor_image = transformTensor(image).to(device)

batch_tensor_image = tensor_image.unsqueeze(0)

proba_before = through_model(batch_tensor_image, model, device)

print(proba_before)

start1 = time.time()

best_pixels_probabilities = best_pixels(tensor_image, True, 1, 0, model, device)

end1 = time.time()

print(end1-start1)


# list_settings = [(0.7, 0.005), (0.65, 0.005), (0.6, 0.005), (0.55, 0.005), (0.5, 0.005), (0.45, 0.005), (0.4, 0.005), (0.35, 0.005), (0.3, 0.005), (0.25, 0.005), (0.2, 0.005)]
list_pourcentages = np.arange(0, 1, 0.05)
list_reach = np.arange(0.005, 0.011, 0.001)

list_probabilities = []

start2 = time.time()

for pr, rc in itertools.product(list_pourcentages, list_reach):

    modified_image = change_pixels(tensor_image, best_pixels_probabilities, pr, rc, device)
    batch_modified_image = modified_image.unsqueeze(0)

    proba_after = through_model(batch_modified_image, model, device)
    proba_after = proba_after[0, 0].item()
    list_probabilities.append((proba_after, pr, rc))
    print(proba_after)

end2 = time.time()

print(end2-start2)

start3 = time.time()

# Extraire les valeurs de probabilité, pourcentage, et reach
probabilities = [item[0] for item in list_probabilities]
pourcentages = sorted(set(item[1] for item in list_probabilities))  # Liste unique de pourcentages
reach = sorted(set(item[2] for item in list_probabilities))  # Liste unique de reach

# Créer une matrice des probabilités en fonction des pourcentages et des reachs
prob_matrix = np.zeros((len(reach), len(pourcentages)))

for prob, pr, rc in list_probabilities:
    pr_idx = pourcentages.index(pr)
    rc_idx = reach.index(rc)
    prob_matrix[rc_idx, pr_idx] = prob

# Création de la heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(prob_matrix, xticklabels=pourcentages, yticklabels=reach, cmap='viridis', annot=True)

# Labels
plt.xlabel('Pourcentage')
plt.ylabel('Reach')
plt.title('Heatmap des Probabilités en fonction de Pourcentage et Reach')
plt.show()

end3 = time.time()

print(end3-start3)