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

image_list = ["../dog_images/0.jpg", "../dog_images/1.jpg", "../dog_images/2.jpg", "../dog_images/3.jpg", "../dog_images/4.jpg"]


results = []


for image_path in image_list:
    print("image suivante")
    result_RGB = ea_base_VF(model, 100, image_path, 40, False, 0, 285, 0.15, 0.03, 0.01, device)
    result_chrome = ea_chrominance_VF_fast(model, 100, image_path, 40, 285, 0.15, 0.09, 0.01, device)

    results.append({
                        "epochs_RGB": result_RGB,
                        "epochs_chrome": result_chrome
                    })


# CSV SECTION

df_results = pd.DataFrame(results)

df_results.to_csv("../csv/YCbCr_results_chrome.csv", index=False)