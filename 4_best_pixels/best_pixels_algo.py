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

def best_pixels_algo(model, tensor_image, wanted_class, pourcentage_total, pourcentage, reach, device):

    # image = Image.open(image_path)
    # image = transformResize(image)
    # tensor_image = transformTensor(image).to(device)


    start = time.time()

    best_pixels_probabilities = best_pixels(tensor_image, True, pourcentage_total, wanted_class, model, device)

    end = time.time()

    print(f'{end-start}s')

    tensor_image = change_pixels(tensor_image, best_pixels_probabilities, pourcentage, reach, device)

    tensor_image = tensor_image.unsqueeze(0)

    probabilities = through_model(tensor_image, model, device)

    probability = probabilities[:, wanted_class]

    print(probability)

    tensor_image = tensor_image.squeeze()

    return tensor_image, probability


def fixed_best_pixels_algo(model, tensor_image, random_pixels, wanted_class, pourcentage, reach, device):

    # image = Image.open(image_path)
    # image = transformResize(image)
    # tensor_image = transformTensor(image).to(device)


    start = time.time()

    best_pixels_probabilities = fixed_best_pixels(tensor_image, random_pixels, wanted_class, model, device)

    end = time.time()

    print(f'{end-start}s')

    tensor_image = change_pixels(tensor_image, best_pixels_probabilities, pourcentage, reach, device)

    tensor_image = tensor_image.unsqueeze(0)

    probabilities = through_model(tensor_image, model, device)

    probability = probabilities[:, wanted_class]

    print(probability)

    tensor_image = tensor_image.squeeze()

    return tensor_image, probability