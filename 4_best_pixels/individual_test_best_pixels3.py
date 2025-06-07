# IMPORTS

from best_pixels_algo import *

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

# TESTING ZONE

image = Image.open( "../dog_images/0.jpg")
image = transformResize(image)
tensor_image = transformTensor(image).to(device)

# STABLE PARAMETERS


start = time.time()

tensor_image, probability = best_pixels_algo(model, tensor_image, 285, 0.25, 1, 0.01, device)

end = time.time()

print(f"Execution time: {end - start}s")

start = time.time()

tensor_image, probability = best_pixels_algo(model, tensor_image, 285, 0.5, 1, 0.01, device)

end = time.time()

print(f"Execution time: {end - start}s")

start = time.time()

tensor_image, probability = best_pixels_algo(model, tensor_image, 285, 1, 1, 0.01, device)

end = time.time()

print(f"Execution time: {end - start}s")

