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
import sys
import math



# VERSION 3
# Like the version 2 but no noise added, rather than that we generate new noise with base images to stay between -reach and reach

def ea_base(model, enums, image_url, batch, targeted, targeted_channel, wanted_class, height, reach, pourcentage, device):
  
  # WITH ADDITIONAL NOISE -> ADS UP

  # Free GPU cache
  torch.cuda.empty_cache()

  elite = int(batch/4)
  mid = int(batch/2)

  # 🧑‍🎨 LOAD AND PROCESS THE BASE IMAGE
  best_probability = 0

  image = Image.open(image_url)
  image = transformResize(image)

  tensor_image = transformTensor(image).to(device)
  multiple_copies = multiple_copies_generator(tensor_image, batch, device)
  base_images = multiple_copies[:mid].clone()

  multiple_copies = noise_generator(multiple_copies, pourcentage, reach, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(multiple_copies, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(multiple_copies, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (20-20) => 40

    top_selection = torch.cat((elite_selection, middle_selection))

    # 20 -> new noises
    noise_selection = noise_generator(multiple_copies, pourcentage, reach, targeted, targeted_channel, device)

    # 20 -> crossover
    parents_index = parent_generator_fixed(top_selection, device)
    crossed_copies = crossover_generator(top_selection, parents_index, height, device)

    # 20 + 20 = 40 index
    multiple_copies = torch.cat((crossed_copies, noise_selection)) # concatenate the splits together

    # ❗️Now we have the final image set for this iteration to pass through the model!

    # 🤖 MODEL AND SELECTION
    probabilities = through_model(multiple_copies, model, device)

    elite_selection, middle_selection, elite_index, elite_proba = selection(multiple_copies, probabilities, elite, wanted_class, device)

    if elite_proba[0] > best_probability:
      best_probability = elite_proba[0]

    
  return best_probability.cpu().item()