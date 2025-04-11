# IMPORTS

from ea import *

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



# INITIALISATION

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model
model_name = "hilmansw/resnet18-catdog-classifier"

model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
model.eval()



# JOB

enums = 100
image_url="../dog.jpg"
batch = 40
blurry = False
blurriness = 8
targeted = False
targeted_channel = 0
wanted_class = 0
height = 0.25
reach = np.arange(0, 0.21, 0.02) # Goes from 0 to 0.2 with a step of 0.02
manual = False
divider = 56
elite_matrices = 0
min_pourcentage = 0.5

best_probabilities = []

for r in reach:

    buffer = ea_edges_v3(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, height, r, 
             manual, divider, elite_matrices, min_pourcentage, device)
    
    best_probabilities.append(buffer)

import matplotlib.pyplot as plt

plt.plot(best_probabilities)
plt.title("evolution of the best probabilities as a function of reach")
plt.xlabel("iterations")
plt.ylabel("probabilities")
plt.grid(True)
plt.show()

    



