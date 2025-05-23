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



image = Image.open("../dog_images/7.jpg")
image = transformResize(image)

tensor_image = transformTensor(image).to(device)

batch_image = multiple_copies_generator(tensor_image, 40, device)

modified_batch_image = noise_generator_edge(batch_image, 0.5, 56, 314, False, 0, False, 0, device)

modified_image = modified_batch_image[0].unsqueeze(0) # Pour le passer dans le model

probability = through_model(modified_image, model, device)

val, idx = torch.max(probability, dim=1)

print(val, idx)

modified_image = modified_image.squeeze()

image_restored = transformPIL(modified_image)

plt.imshow(image_restored)
plt.axis("off")
plt.show()