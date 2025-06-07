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

model_name = 'resnet50.tv_in1k'
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
    wanted_class = 285
    pourcentage_fixed = 0.01
    height = 0.15
    reach = 0.03
    manual = False
    divider = 56
    elite_matrices = 20
    min_pourcentage = 0.5

    # CHANGING PARAMETERS

    changing_divider = [8, 16, 32, 56]

    changing_min_pourcentage = [
        6/64,   # 0.09375 ~ 0.1
        13/64,  # 0.203125 ~ 0.2
        26/64,  # 0.40625 ~ 0.4
        39/64,  # 0.609375 ~ 0.6
        51/64,  # 0.796875 ~ 0.8
        57/64,  # 0.890625 ~ 0.9
        61/64,  # 0.953125 ~ 0.95
        64/64   # 1.0
    ]

    # COMPUTE SECTION
    results = []

    start = time.time()

    for image_url in image_list:
        for cd, cmp in itertools.product(changing_divider, changing_min_pourcentage):

            pourcentage = (pourcentage_fixed/cmp)

            result_before = ea_edges_VF(model, 1, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, pourcentage, height, reach, 
                    manual, cd, elite_matrices, cmp, device)
            
            result_after = ea_edges_VF(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, pourcentage, height, reach, 
                    manual, cd, elite_matrices, cmp, device)
            
            result_evolution = result_after - result_before
            
            results.append({
                                    "model": model_name,
                                    "image": image_url,
                                    "divider": cd,
                                    "min_pourcentage": cmp,
                                    "result_evolution": result_evolution
                                })
            
            # results.append({
            #                         "model": model_name,
            #                         "image": image_url,
            #                         "blurry": blurry,
            #                         "blurriness": blurriness,
            #                         "targeted": targeted,
            #                         "targeted_channel": targeted_channel,
            #                         "pourcentage": pourcentage,
            #                         "height": height,
            #                         "reach": reach,
            #                         "divider": cd,
            #                         "min_pourcentage": cmp,
            #                         "result_evolution": result_evolution
            #                     })

    end = time.time()

    print(f"Execution time: {end - start}s")

    return results


# CSV SECTION

final = table_maker(model_name, device, 100, image_list, False, 0, False, 0)

df_results = pd.DataFrame(final)

df_results.to_csv("../csv/edge_results.csv", index=False)