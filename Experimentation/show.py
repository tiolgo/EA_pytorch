
# IMPORTS

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
import sys
import math

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

image = Image.open("../dog_images/1.jpg")
image = transformResize(image)

tensor_image = transformTensor(image).to(device)
neutral_tensor = torch.full((3, 224, 224), 0.1, device='cuda')
neutral_tensor_batch = multiple_copies_generator(neutral_tensor, 40, device)
noise_tensor = noise_generator(neutral_tensor_batch, 1, 0.5, False, 0, device)
combined_images = (noise_tensor[0] + tensor_image) / 2

image_restored = transformPIL(combined_images)
plt.imshow(image_restored)
plt.axis("off")
plt.show()