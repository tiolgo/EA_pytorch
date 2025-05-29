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

def best_pixels_algo(model, max_epoch, image_path, wanted_class, pourcentage, reach, device):

    image = Image.open(image_path)
    image = transformResize(image)
    tensor_image = transformTensor(image).to(device)


    start = time.time()

    best_pixels_probabilities = best_pixels(tensor_image, False, 1, wanted_class, model, device)

    end = time.time()

    print(f'{end-start}s')

    for _ in range(max_epoch):

        modified_image = change_pixels(tensor_image, best_pixels_probabilities, pourcentage, reach, device)

        modified_image = modified_image.unsqueeze(0)

        probabilities = through_model(modified_image, model, device)

        probability = probabilities[:, wanted_class]

        print(probability)

        modified_image = modified_image.squeeze()

    image_restored = transformPIL(modified_image)

    plt.imshow(image_restored)
    plt.axis("off")
    plt.show()