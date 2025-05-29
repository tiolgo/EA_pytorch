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
model = timm.create_model('resnet18.tv_in1k', pretrained=True)
model = model.to(device)
model = model.eval()

reach = 0.03

image = Image.open("../dog_images/0.jpg")
image = transformResize(image)

tensor_image = transformTensor(image).to(device)

base_image = tensor_image.clone()

tensor_image = torch.full((3, 224, 224), reach, device='cuda')

tensor_image = tensor_image.unsqueeze(0)

tensor_image_mutated = noise_generator(tensor_image, 1, reach, False, 0, device)

probability = through_model(tensor_image_mutated, model, device)
 
val, idx = torch.max(probability, dim=1)

print(val, idx)

modified_image = tensor_image_mutated.squeeze()

combined_image = modified_image + (base_image * (1 - 2 * reach))

combined_image = combined_image.unsqueeze(0)

probability = through_model(combined_image, model, device)
 
val, idx = torch.max(probability, dim=1)

print(val, idx)

combined_image = combined_image.squeeze()

image_restored = transformPIL(combined_image)

plt.imshow(image_restored)
plt.axis("off")
plt.show()