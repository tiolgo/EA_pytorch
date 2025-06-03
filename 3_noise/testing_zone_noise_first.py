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

model_name = 'resnet50.tv_in1k'

model = timm.create_model(model_name, pretrained=True)
model = model.to(device)
model = model.eval()


# TESTING ZONE

image_list = ["../dog_images/0.jpg", "../dog_images/1.jpg", "../dog_images/2.jpg", "../dog_images/3.jpg", "../dog_images/4.jpg", "../dog_images/5.jpg",
              "../dog_images/6.jpg", "../dog_images/7.jpg", "../dog_images/8.jpg", "../dog_images/9.jpg"]

reach = [0.01, 0.03, 0.05, 0.07, 0.09]

results = []

for _ in range(2):

    mask, mask_probability = ea_noise_image(model, 10000, 40, False, 0, 285, 0.15, 0.07, 0.01, 0.9, device)

    for image_path in image_list:

        image = Image.open(image_path)
        image = transformResize(image)

        tensor_image = transformTensor(image).to(device)

        combined_image = mask + (tensor_image * (1 - 2 * 0.07))

        combined_image = combined_image.unsqueeze(0)

        probabilities1 = through_model(combined_image, model, device)

        val1, idx1 = torch.max(probabilities1, dim=1)

        tensor_image = tensor_image.unsqueeze(0)

        probabilities2 = through_model(tensor_image, model, device)

        val2, idx2 = torch.max(probabilities2, dim=1)

        
        results.append({
                            "image_path": image_path,
                            "mask_probability": mask_probability,
                            "best_value_base": val2,
                            "best_label_base": idx2,
                            "best_value_mask": val1,
                            "best_label_mask": idx1,
                            
                        })


df_results = pd.DataFrame(results)

df_results.to_csv("../csv/noise_results_third.csv", index=False)

