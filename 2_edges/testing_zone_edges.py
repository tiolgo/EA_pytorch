# IMPORTS

from ea_edges import *

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

model = timm.create_model('vgg16.tv_in1k', pretrained=True)
model = model.to(device)
model = model.eval()

model_list = ['vgg16.tv_in1k', 'vgg19.tv_in1k', 'resnet50.tv_in1k', 'resnet101.tv_in1k', 'resnet152.tv_in1k', 'densenet121.tv_in1k', 'densenet169.tv_in1k', 'densenet201.tv_in1k']
image_list = ["../dog_images/0.jpg", "../dog_images/1.jpg", "../dog_images/2.jpg", "../dog_images/3.jpg", "../dog_images/4.jpg", "../dog_images/5.jpg",
              "../dog_images/6.jpg", "../dog_images/7.jpg", "../dog_images/8.jpg", "../dog_images/9.jpg"]



# TESTING ZONE

def table_maker(model_name, device, enums, image_list, blurry, blurriness, targeted, targeted_channel):

    # Set model
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device)
    model = model.eval()

    # STABLE PARAMETERS
    enums = enums
    batch = 40
    blurry = blurry
    blurriness = blurriness
    targeted = targeted
    targeted_channel = targeted_channel
    wanted_class = 0
    height = 0.15
    reach = 0.1
    manual = False
    divider = 7
    elite_matrices = 20
    min_pourcentage = 0.5

    # CHANGING PARAMETERS
    changing_height = np.arange(0, 1.05, 0.05) # 20
    changing_reach = np.arange(0, 0.105, 0.005) # 20
    changing_divider = [1, 2, 4, 7, 8, 14, 16, 28, 32, 56] # 10 56 is th maximum because de bluriness is set to 4
    changing_min_pourcentage = np.arange(0, 1.05, 0.05) # 20

    # HEIGHT SECTION
    results = []

    start = time.time()

    for image_url in image_list:
        for ch in changing_height:
            result = ea_edges_v3(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, ch, reach, 
                    manual, divider, elite_matrices, min_pourcentage, device)
            
            results.append({
                                    "model": model_name,
                                    "image": image_url,
                                    "blurry": blurry,
                                    "blurriness": blurriness,
                                    "targeted": targeted,
                                    "targeted_channel": targeted_channel,
                                    "height": ch,
                                    "reach": reach,
                                    "divider": divider,
                                    "min_pourcentage": min_pourcentage,
                                    "result": result
                                })

    end = time.time()

    print(f"Execution time: {end - start}s")

    # REACH SECTION

    start = time.time()

    for image_url in image_list:
        for cr in changing_reach:
            result = ea_edges_v3(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, height, cr, 
                    manual, divider, elite_matrices, min_pourcentage, device)
            
            results.append({
                                    "model": model_name,
                                    "image": image_url,
                                    "blurry": blurry,
                                    "blurriness": blurriness,
                                    "targeted": targeted,
                                    "targeted_channel": targeted_channel,
                                    "height": height,
                                    "reach": cr,
                                    "divider": divider,
                                    "min_pourcentage": min_pourcentage,
                                    "result": result
                            })

    end = time.time()

    print(f"Execution time: {end - start}s")

    # DIVIDER SECTION

    start = time.time()

    for image_url in image_list:
        for cd in changing_divider:
            result = ea_edges_v3(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, height, reach, 
                    manual, cd, elite_matrices, min_pourcentage, device)
            
            results.append({
                                    "model": model_name,
                                    "image": image_url,
                                    "blurry": blurry,
                                    "blurriness": blurriness,
                                    "targeted": targeted,
                                    "targeted_channel": targeted_channel,
                                    "height": height,
                                    "reach": reach,
                                    "divider": cd,
                                    "min_pourcentage": min_pourcentage,
                                    "result": result
                                })

    end = time.time()

    print(f"Execution time: {end - start}s")

    # MIN POURCENTAGE SECTION

    start = time.time()

    for image_url in image_list:
        for cmp in changing_min_pourcentage:
            result = ea_edges_v3(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, height, reach, 
                    manual, divider, elite_matrices, cmp, device)
            
            results.append({
                                    "model": model_name,
                                    "image": image_url,
                                    "blurry": blurry,
                                    "blurriness": blurriness,
                                    "targeted": targeted,
                                    "targeted_channel": targeted_channel,
                                    "height": height,
                                    "reach": reach,
                                    "divider": divider,
                                    "min_pourcentage": cmp,
                                    "result": result
                                })

    end = time.time()

    print(f"Execution time: {end - start}s")

    return results



for model_name in model_list:
    a = table_maker(model_name, device, 100, image_list, False, 0, False, 0)
    b = table_maker(model_name, device, 100, image_list, False, 0, True, 0)
    c = table_maker(model_name, device, 100, image_list, False, 0, True, 1)
    d = table_maker(model_name, device, 100, image_list, False, 0, True, 2)
    e = table_maker(model_name, device, 100, image_list, True, 4, False, 0)

    final = a + b + c + d + e
    df_results = pd.DataFrame(final)

    df_results.to_csv("../csv/results.csv", index=False)