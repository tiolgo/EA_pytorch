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

image = Image.open( "../dog_images/1.jpg")
image = transformResize(image)
tensor_image = transformTensor(image).to(device)

results = []


# STABLE PARAMETERS

for i in range(10):

    tensor_image, probability = best_pixels_algo(model, tensor_image, 587, 0.5, 0.9, 0.01, device)
    results.append({
                        "epoch": i,
                        "probability": probability,
                    })
    

# CSV SECTION

# df_results = pd.DataFrame(results)

# df_results.to_csv("../csv/best_pixels_results_25_0.005_rdm.csv", index=False)