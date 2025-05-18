# IMPORTS

from Experimentation.ea_edges import *

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
import time
from mpl_toolkits.mplot3d import Axes3D



# INITIALISATION

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set model
model_name = "hilmansw/resnet18-catdog-classifier"

model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
model.eval()



image = Image.open("../dog_images/1.jpg")
image = transformResize(image)

tensor_image = transformTensor(image).to(device)

start1 = time.time()

best_pixels_probabilities = best_pixels(tensor_image, False, 0.01, 0, model, device)

end1 = time.time()

print(end1-start1)

modified_image = change_pixels(tensor_image, best_pixels_probabilities, 0.6, 0.006, device)
batch_modified_image = modified_image.unsqueeze(0)
proba_after = through_model(batch_modified_image, model, device)
proba_after = proba_after[0, 0].item()
print(proba_after)

image_restored = transformPIL(modified_image)
plt.imshow(image_restored)
plt.axis("off")
plt.show()