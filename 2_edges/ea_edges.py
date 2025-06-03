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
import time



# EVOLUTIONARY ALGORITHMS

def ea_edges_VF(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, pourcentage, height, reach, 
             manual, divider, elite_matrices, min_pourcentage, device):
  
  # WITH ADDITIONAL NOISE -> ADS UP

  # Free GPU cache
  torch.cuda.empty_cache()

  elite = int(batch/4)
  mid = int(batch/2)

  if not manual:

    elite_matrices = 0

    while elite_matrices/divider**2 < min_pourcentage:
      elite_matrices += 1

  # 🧑‍🎨 LOAD AND PROCESS THE BASE IMAGE
  best_probability = 0

  image = Image.open(image_url)
  image = transformResize(image)

  tensor_image = transformTensor(image).to(device)
  multiple_copies = multiple_copies_generator(tensor_image, batch, device)
  base_images = multiple_copies.clone()

  coord_tensor = find_best_patch(multiple_copies, divider, elite_matrices, targeted, targeted_channel, device)

  base_selection = noise_generator_patch(base_images, coord_tensor, pourcentage, reach, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(base_selection, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (10-30) => 40

    # Dont touche the elite

    # Mutate the middle
    middle_mutated = noise_generator_patch(middle_selection, coord_tensor, pourcentage, reach, device)
    middle_mutated = torch.clamp(middle_mutated, base_images[:elite] - reach, base_images[:elite] + reach)

    # elite + 10 random images
    rand_indices = torch.randperm(batch)[:elite]
    random_selection = base_selection[rand_indices]
    keep_selection = torch.cat((elite_selection, random_selection))
    keep_mutated = noise_generator_patch(keep_selection, coord_tensor, pourcentage, reach, device)
    keep_mutated = torch.clamp(keep_mutated, base_images[:mid] - reach, base_images[:mid] + reach)
    keep_mutated = noise_generator_patch(keep_mutated, coord_tensor, pourcentage, reach, device)
    keep_mutated = torch.clamp(keep_mutated, base_images[:mid] - reach, base_images[:mid] + reach)

    base_selection = torch.cat((elite_selection, middle_mutated, keep_mutated)) # concatenate the splits together

    probabilities = through_model(base_selection, model, device)

    elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

    # 30 -> crossover

    top_selection = torch.cat((middle_mutated, keep_mutated))
    parents_index = parent_generator_fixed(top_selection, device)
    crossed_selection = crossover_generator(top_selection, parents_index, height, device)

    # 10 + 30 = 40 index
    base_selection = torch.cat((elite_selection, crossed_selection)) # concatenate the splits together

    # ❗️Now we have the final image set for this iteration to pass through the model!

    # 🤖 MODEL AND SELECTION

    probabilities = through_model(base_selection, model, device)

    elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

    if elite_proba[0] > best_probability:
      best_probability = elite_proba[0]

    
  return best_probability.cpu().item()



def ea_edges_random(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, pourcentage, height, reach, 
             manual, divider, elite_matrices, min_pourcentage, device):
  
  # WITH ADDITIONAL NOISE -> ADS UP

  # Free GPU cache
  torch.cuda.empty_cache()

  elite = int(batch/4)
  mid = int(batch/2)

  if not manual:

    elite_matrices = 0

    while elite_matrices/divider**2 < min_pourcentage:
      elite_matrices += 1

  # 🧑‍🎨 LOAD AND PROCESS THE BASE IMAGE
  best_probability = 0

  image = Image.open(image_url)
  image = transformResize(image)

  tensor_image = transformTensor(image).to(device)
  multiple_copies = multiple_copies_generator(tensor_image, batch, device)
  base_images = multiple_copies.clone()

  coord_tensor = find_random_patch(multiple_copies, divider, elite_matrices, targeted, targeted_channel, device)

  base_selection = noise_generator_patch(base_images, coord_tensor, pourcentage, reach, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(base_selection, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (10-30) => 40

    # Dont touche the elite

    # Mutate the middle
    middle_mutated = noise_generator_patch(middle_selection, coord_tensor, pourcentage, reach, device)
    middle_mutated = torch.clamp(middle_mutated, base_images[:elite] - reach, base_images[:elite] + reach)

    # elite + 10 random images
    rand_indices = torch.randperm(batch)[:elite]
    random_selection = base_selection[rand_indices]
    keep_selection = torch.cat((elite_selection, random_selection))
    keep_mutated = noise_generator_patch(keep_selection, coord_tensor, pourcentage, reach, device)
    keep_mutated = torch.clamp(keep_mutated, base_images[:mid] - reach, base_images[:mid] + reach)
    keep_mutated = noise_generator_patch(keep_mutated, coord_tensor, pourcentage, reach, device)
    keep_mutated = torch.clamp(keep_mutated, base_images[:mid] - reach, base_images[:mid] + reach)

    base_selection = torch.cat((elite_selection, middle_mutated, keep_mutated)) # concatenate the splits together

    probabilities = through_model(base_selection, model, device)

    elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

    # 30 -> crossover

    top_selection = torch.cat((middle_mutated, keep_mutated))
    parents_index = parent_generator_fixed(top_selection, device)
    crossed_selection = crossover_generator(top_selection, parents_index, height, device)

    # 10 + 30 = 40 index
    base_selection = torch.cat((elite_selection, crossed_selection)) # concatenate the splits together

    # ❗️Now we have the final image set for this iteration to pass through the model!

    # 🤖 MODEL AND SELECTION

    probabilities = through_model(base_selection, model, device)

    elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

    if elite_proba[0] > best_probability:
      best_probability = elite_proba[0]

    
  return best_probability.cpu().item()



def ea_edges_v4(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, height, reach, 
             manual, divider, elite_matrices, min_pourcentage, device):
  
  # WITH ADDITIONAL NOISE -> ADS UP

  # Free GPU cache
  torch.cuda.empty_cache()

  elite = int(batch/4)
  mid = int(batch/2)

  if not manual:

    elite_matrices = 0

    while elite_matrices/divider**2 < min_pourcentage:
      elite_matrices += 1

  # 🧑‍🎨 LOAD AND PROCESS THE BASE IMAGE
  best_probability = 0

  image = Image.open(image_url)
  image = transformResize(image)

  tensor_image = transformTensor(image).to(device)
  multiple_copies = multiple_copies_generator(tensor_image, batch, device)
  base_images = multiple_copies[:mid].clone()

  coord_tensor = find_best_patch(multiple_copies, divider, elite_matrices, targeted, targeted_channel, device)

  multiple_copies = noise_generator_patch(multiple_copies, coord_tensor, 1, reach, device)

  

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(multiple_copies, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(multiple_copies, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (20-20) => 40

    top_selection = torch.cat((elite_selection, middle_selection))

    # 20 -> new noises
    noise_selection = noise_generator_patch(base_images, coord_tensor, 1, reach, device)

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