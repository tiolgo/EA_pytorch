# IMPORTS
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../1_base')))

from ea_base import *
from ea_YCbCr import *

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

result = ea_base_image(model, 1000, "../dog_images/0.jpg", 40, True, 2, 285, 0.15, 0.2, 0.01, 0.7, device)

end = time.time()

print(f"Execution time: {end - start}s")

image_restored = transformPIL(result)

plt.imshow(image_restored)
plt.axis("off")
plt.show()


start = time.time()

result = ea_YCbCr_image(model, 1000, "../dog_images/0.jpg", 40, True, 1, 285, 0.15, 0.2, 0.01, 0.7, device)

end = time.time()

print(f"Execution time: {end - start}s")

result = result.unsqueeze(0)
result = from_ycbcr_to_rgb_batch(result, device)
result = result.squeeze()

image_restored = transformPIL(result)

plt.imshow(image_restored)
plt.axis("off")
plt.show()

start = time.time()

result = ea_YCbCr_image(model, 1000, "../dog_images/0.jpg", 40, True, 2, 285, 0.15, 0.2, 0.01, 0.7, device)

end = time.time()

print(f"Execution time: {end - start}s")

result = result.unsqueeze(0)
result = from_ycbcr_to_rgb_batch(result, device)
result = result.squeeze()

image_restored = transformPIL(result)

plt.imshow(image_restored)
plt.axis("off")
plt.show()