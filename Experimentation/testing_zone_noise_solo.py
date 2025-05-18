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
model = timm.create_model('densenet121.tv_in1k', pretrained=True)
model = model.to(device)
model = model.eval()

# STABLE PARAMETERS
enums = 1000
image_url = "../dog_images/5.jpg"
batch = 40
wanted_class = 285
height = 0.15
reach = 0.5
pourcentage = 1

start = time.time()

result = ea_noise(model, enums, image_url, batch, wanted_class, height, reach, pourcentage, device)
    
end = time.time()

print(f"Execution time: {end - start}s")
print(result)