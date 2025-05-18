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



image = Image.open("../dog_images/1.jpg")
image = transformResize(image)

tensor_image = transformTensor(image).to(device)

zeros = torch.zeros_like(tensor_image[0])  # [H, W]
tensor_image_R = torch.stack([tensor_image[2], zeros, zeros])
tensor_image_G = torch.stack([zeros, tensor_image[2], zeros])
tensor_image_B = torch.stack([zeros, zeros, tensor_image[2]])

image_restored = transformPIL(tensor_image_B)

plt.imshow(image_restored)
plt.axis("off")
plt.show()