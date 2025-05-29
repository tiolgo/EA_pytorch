# IMPORTS
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../1_base')))

from ea_base import *

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

model_name = 'resnet50.tv_in1k'

model = timm.create_model(model_name, pretrained=True)
model = model.to(device)
model = model.eval()


# TESTING ZONE

image_list = ["../dog_images/0.jpg", "../dog_images/1.jpg", "../dog_images/2.jpg", "../dog_images/3.jpg", "../dog_images/4.jpg"]

results = []

for image_path in image_list:
    mean = 0
    for _ in range(2):
        mean += ea_base_epoch(model, 10000, image_path, 40, False, 0, 285, 0.15, 0.03, 0.01, 0.1, device)
    
    mean = mean/2

    results.append({
                        "mean": mean
                    })



mean = 0

for _ in range(2):
    mean += ea_noise_epoch(model, 10000, 40, False, 0, 285, 0.15, 0.03, 0.01, 0.1, device)

mean = mean/2

results.append({
                    "mean": mean
                })


df_results = pd.DataFrame(results)

df_results.to_csv("../csv/small_noise_results_debug.csv", index=False)

