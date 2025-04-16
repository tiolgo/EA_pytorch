# IMPORTS

from ea import *

from transformers import AutoModelForImageClassification, AutoProcessor, AutoImageProcessor
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import requests
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
import math
import itertools



# INITIALISATION

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model
model_name = "hilmansw/resnet18-catdog-classifier"

model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
model.eval()



# TESTING ZONE

# Test noise_generator_edge()

# image = Image.open("../dog_images/9.jpg")
# image = transformResize(image)
# tensor_image = transformTensor(image).to(device)

# group_tensor_image = multiple_copies_generator(tensor_image, 40, device)
# print(group_tensor_image.shape)

# group_tensor_image = noise_generator_edge(group_tensor_image, reach = 1, divider = 56, elite_matrices=400, blurry=False, blurriness=2, targeted=False, targeted_channel=0, device=device)
# test_image = group_tensor_image[0]

# image_restored = transformPIL(test_image)
# plt.imshow(image_restored)
# plt.axis("off")
# plt.show()



image = Image.open("../dog_images/0.jpg")
image = transformResize(image)
tensor_image = transformTensor(image).to(device)

sign, proba = individual_pixel(model, tensor_image, 0, 0, 0, 0, device)