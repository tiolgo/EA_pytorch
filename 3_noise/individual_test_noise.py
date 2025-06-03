# IMPORTS

from ea_noise import *

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


# STABLE PARAMETERS

start = time.time()

result = ea_noise_mask_only(model, 10000, 40, False, 0, 285, 0.15, 0.06, 0.01, device)

end = time.time()

print(f"Execution time: {end - start}s")

print(result)

start = time.time()

result = ea_noise_mask_only(model, 10000, 40, False, 0, 296, 0.15, 0.06, 0.01, device)

end = time.time()

print(f"Execution time: {end - start}s")

print(result)

start = time.time()

result = ea_noise_mask_only(model, 10000, 40, False, 0, 621, 0.15, 0.06, 0.01, device)

end = time.time()

print(f"Execution time: {end - start}s")

print(result)

start = time.time()

result = ea_noise_mask_only(model, 10000, 40, False, 0, 13, 0.15, 0.06, 0.01, device)

end = time.time()

print(f"Execution time: {end - start}s")

print(result)

start = time.time()

result = ea_noise_mask_only(model, 10000, 40, False, 0, 621, 0.15, 0.25, 0.01, device)

end = time.time()

print(f"Execution time: {end - start}s")

print(result)

start = time.time()

result = ea_noise_mask_only(model, 10000, 40, False, 0, 621, 0.15, 0.5, 0.01, device)

end = time.time()

print(f"Execution time: {end - start}s")

print(result)