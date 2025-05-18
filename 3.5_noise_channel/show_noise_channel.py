# IMPORTS

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import *

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

targeted_channel = 0

image = Image.open("../dog_images/1.jpg")
image = transformResize(image)

tensor_image = transformTensor(image).to(device)

tensor_zeros = torch.zeros((3, 224, 224))
tensor_zeros[targeted_channel] = 0.5

batch_image = multiple_copies_generator(tensor_zeros, 40, device)

modified_batch_image = noise_generator(batch_image, 1, 0.1, True, targeted_channel, device)

modified_image = modified_batch_image[0]

combined_image = tensor_image.clone() # Pour le passer dans le model

combined_image[targeted_channel] = modified_image[targeted_channel]

combined_image = combined_image.unsqueeze(0)

probability = through_model(combined_image, model, device)

val, idx = torch.max(probability, dim=1)

print(val, idx)

combined_image = combined_image.squeeze()

image_restored = transformPIL(combined_image)

plt.imshow(image_restored)
plt.axis("off")
plt.show()