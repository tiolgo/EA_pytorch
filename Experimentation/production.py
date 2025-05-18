# IMPORTS

from Experimentation.ea_edges import *

from transformers import AutoModelForImageClassification, AutoProcessor, AutoImageProcessor
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import sys
import math
import itertools


# INITIALISATION

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model
model_name = "hilmansw/resnet18-catdog-classifier"

model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
model.eval()



# JOB

enums = 100
image_url="../dog_images/0.jpg"
batch = 40
wanted_class = 0

blurry = False
blurriness = 8

targeted = False
targeted_channel = 0

manual = False
divider = 56
elite_matrices = 0

height = np.arange(0, 0.51, 0.05) # x10
reach = np.arange(0, 0.21, 0.02) # x10
min_pourcentage = np.arange(0, 1.01, 0.1) # x10

best_probabilities = []
coordinates = []

for mp, r, h in itertools.product(min_pourcentage, reach, height):

    best_probability = ea_edges_v3(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, h, r,
             manual, divider, elite_matrices, mp, device)
    print(f"h={h}; r={r}; mp={mp}; best_probability={best_probability}")

    best_probabilities.append(best_probability)
    coordinates.append((mp, r, h))
mp, r, h = zip(*coordinates)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(mp, r, h, c=h, cmap='viridis', s=100)

fig.colorbar(scatter, ax=ax, label='h value')

ax.set_xlabel('mp')
ax.set_ylabel('r')
ax.set_zlabel('h')
ax.set_title('adv image 3D plot')

plt.show()
    



