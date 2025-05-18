# IMPORTS

from Experimentation.ea_edges import *

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

# # Set model
# model_name = "hilmansw/resnet18-catdog-classifier"

# model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
# model.eval()

model = timm.create_model('densenet121.tv_in1k', pretrained=True)
model = model.to(device)
model = model.eval()

# STABLE PARAMETERS
enums = 100
image_url = "../dog_images/5.jpg"
batch = 40
blurry = False
blurriness = 2
targeted = False
targeted_channel = 0
wanted_class = 285
height = 0.15
reach = 0.1
manual = False
divider = 7
elite_matrices = 20
min_pourcentage = 0.1

start = time.time()

result = ea_edges_v3(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, height, reach, 
        manual, divider, elite_matrices, min_pourcentage, device)
    

end = time.time()

print(f"Execution time: {end - start}s")
print(result)