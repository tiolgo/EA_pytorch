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

# Set model
model = timm.create_model(model_name, pretrained=True)
model = model.to(device)
model = model.eval()

results = []

pourcentage_fixed = 0.01



min_pourcentage = 57/64

pourcentage = (pourcentage_fixed/min_pourcentage)

result_targeted = ea_edges_VF(model, 1000, "../dog_images/2.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 32, 50, min_pourcentage, device)

result_untargeted = ea_edges_VF(model, 1000, "../dog_images/2.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 32, 50, 1, device)


results.append({
                        "image": "../dog_images/2.jpg",
                        "result_targeted": result_targeted,
                        "result_untargeted": result_untargeted,
                    })


min_pourcentage = 61/64

pourcentage = (pourcentage_fixed/min_pourcentage)

result_targeted = ea_edges_VF(model, 1000, "../dog_images/3.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 56, 50, min_pourcentage, device)

result_untargeted = ea_edges_VF(model, 1000, "../dog_images/3.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 56, 50, 1, device)


results.append({
                        "image": "../dog_images/3.jpg",
                        "result_targeted": result_targeted,
                        "result_untargeted": result_untargeted,
                    })





min_pourcentage = 39/64

pourcentage = (pourcentage_fixed/min_pourcentage)

result_targeted = ea_edges_VF(model, 1000, "../dog_images/4.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 32, 50, min_pourcentage, device)

result_untargeted = ea_edges_VF(model, 1000, "../dog_images/4.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 32, 50, 1, device)


results.append({
                        "image": "../dog_images/4.jpg",
                        "result_targeted": result_targeted,
                        "result_untargeted": result_untargeted,
                    })
    


min_pourcentage = 26/64

pourcentage = (pourcentage_fixed/min_pourcentage)

result_targeted = ea_edges_VF(model, 1000, "../dog_images/5.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 8, 50, min_pourcentage, device)

result_untargeted = ea_edges_VF(model, 1000, "../dog_images/5.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 8, 50, 1, device)


results.append({
                        "image": "../dog_images/5.jpg",
                        "result_targeted": result_targeted,
                        "result_untargeted": result_untargeted,
                    })
    

min_pourcentage = 51/64

pourcentage = (pourcentage_fixed/min_pourcentage)

result_targeted = ea_edges_VF(model, 1000, "../dog_images/6.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 32, 50, min_pourcentage, device)

result_untargeted = ea_edges_VF(model, 1000, "../dog_images/6.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 32, 50, 1, device)


results.append({
                        "image": "../dog_images/6.jpg",
                        "result_targeted": result_targeted,
                        "result_untargeted": result_untargeted,
                    })


min_pourcentage = 61/64

pourcentage = (pourcentage_fixed/min_pourcentage)

result_targeted = ea_edges_VF(model, 1000, "../dog_images/8.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 8, 50, min_pourcentage, device)

result_untargeted = ea_edges_VF(model, 1000, "../dog_images/8.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 8, 50, 1, device)


results.append({
                        "image": "../dog_images/8.jpg",
                        "result_targeted": result_targeted,
                        "result_untargeted": result_untargeted,
                    })


min_pourcentage = 51/64

pourcentage = (pourcentage_fixed/min_pourcentage)

result_targeted = ea_edges_VF(model, 1000, "../dog_images/9.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 8, 50, min_pourcentage, device)

result_untargeted = ea_edges_VF(model, 1000, "../dog_images/9.jpg", 40, False, 0, False, 0, 285, pourcentage, 0.15, 0.03, 
        False, 8, 50, 1, device)


results.append({
                        "image": "../dog_images/9.jpg",
                        "result_targeted": result_targeted,
                        "result_untargeted": result_untargeted,
                    })





# CSV SECTION

df_results = pd.DataFrame(results)

df_results.to_csv("../csv/edge_results_second.csv", index=False)