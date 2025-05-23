
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

def ea_noise(model, enums, image_url, batch, wanted_class, height, reach, pourcentage, pourcentage_bruit, device):
  
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
  base_images = multiple_copies_generator(tensor_image, batch, device)

  neutral_tensor = torch.full((3, 224, 224), 0.5, device='cuda')
  batch_neutral_tensor = multiple_copies_generator(neutral_tensor, batch, device)
  batch_neutral_tensor_mid = multiple_copies_generator(neutral_tensor, mid, device)

  noise_copies = noise_generator(batch_neutral_tensor, pourcentage, reach, False, 0, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(noise_copies, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(noise_copies, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (20-20) => 40

    top_selection = torch.cat((elite_selection, middle_selection))

    # 20 -> new noises
    noise_selection = noise_generator(batch_neutral_tensor_mid, pourcentage, reach, False, 0, device)

    # 20 -> crossover
    parents_index = parent_generator_fixed(top_selection, device)
    crossed_copies = crossover_generator(top_selection, parents_index, height, device)

    # 20 + 20 = 40 index
    noise_copies = torch.cat((crossed_copies, noise_selection)) # concatenate the splits together

    # ❗️Now we have the final image set for this iteration to pass through the model!

    # 🤖 MODEL AND SELECTION
    
    probabilities = through_model(noise_copies, model, device)
    elite_selection, middle_selection, elite_index, elite_proba = selection(noise_copies, probabilities, elite, wanted_class, device)

    # combined_images = (noise_copies + base_images) / 2
    combined_images = (noise_copies * pourcentage_bruit) + (base_images * (1 - pourcentage_bruit))
    probabilities_combined = through_model(combined_images, model, device)
    elite_selection_combined, middle_selection_combined, elite_index_combined, elite_proba_combined = selection(combined_images, probabilities_combined, elite, wanted_class, device)
    

    if elite_proba_combined[0] > best_probability:
      best_probability = elite_proba[0]

    
  return best_probability.cpu().item()
