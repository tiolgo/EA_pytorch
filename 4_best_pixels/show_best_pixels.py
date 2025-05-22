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
import pandas as pd
import sys
import math
import itertools
import time
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import timm



# INITIALISATION

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set model
model = timm.create_model('resnet50.tv_in1k', pretrained=True)
model = model.to(device)
model = model.eval()



image = Image.open("../dog_images/1.jpg")

image = transformResize(image)

tensor_image = transformTensor(image).to(device)

tensor_image = tensor_image.unsqueeze(0)

probability = through_model(tensor_image, model, device)

tensor_image = tensor_image.squeeze()

val_before, idx_before = torch.max(probability, dim=1)

print(val_before, idx_before)


best_pixels_probabilities = best_pixels(tensor_image, True, 0.0002, 286, model, device)

modified_image = change_pixels(tensor_image, best_pixels_probabilities, 1, 0.05, device) # soucis

modified_image = modified_image.unsqueeze(0)

probability = through_model(modified_image, model, device)

val_after, idx_after = torch.max(probability, dim=1)

print(val_after, idx_after)

modified_image = modified_image.squeeze()

image_restored = transformPIL(modified_image)

plt.imshow(image_restored)
plt.axis("off")
plt.show()