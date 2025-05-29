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

# EVOLUTIONARY ALGORITHMS

def ea_base_VF(model, enums, image_url, batch, targeted, targeted_channel, wanted_class, height, reach, pourcentage, device):
  
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
  base_images = multiple_copies.clone()

  base_selection = noise_generator(multiple_copies, pourcentage, reach, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(base_selection, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (10-30) => 40

    # Dont touche the elite

    # Mutate the middle
    middle_mutated = noise_generator(middle_selection, pourcentage, reach, targeted, targeted_channel, device)
    middle_mutated = torch.clamp(middle_mutated, base_images[:elite] - reach, base_images[:elite] + reach)

    # elite + 10 random images
    rand_indices = torch.randperm(batch)[:elite]
    random_selection = base_selection[rand_indices]
    keep_selection = torch.cat((elite_selection, random_selection))
    keep_mutated = noise_generator(keep_selection, pourcentage, reach, targeted, targeted_channel, device)
    keep_mutated = torch.clamp(keep_mutated, base_images[:mid] - reach, base_images[:mid] + reach)
    keep_mutated = noise_generator(keep_mutated, pourcentage, reach, targeted, targeted_channel, device)
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


def ea_base_colab(model, enums, image_url, batch, targeted, targeted_channel, wanted_class, height, reach, pourcentage, device):
  
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
  base_images = multiple_copies.clone()

  base_selection = noise_generator(multiple_copies, pourcentage, reach, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(base_selection, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (10-30) => 40

    # Dont touche the elite

    # Mutate the middle
    middle_mutated = noise_generator(middle_selection, pourcentage, reach, targeted, targeted_channel, device)

    # elite + 10 random images
    rand_indices = torch.randperm(batch)[:elite]
    random_selection = base_selection[rand_indices]
    keep_selection = torch.cat((elite_selection, random_selection))
    keep_mutated = noise_generator(keep_selection, pourcentage, reach, targeted, targeted_channel, device)
    keep_mutated = noise_generator(keep_mutated, pourcentage, reach, targeted, targeted_channel, device)

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


def ea_base_epoch(model, enums, image_url, batch, targeted, targeted_channel, wanted_class, height, reach, pourcentage, targeted_probability, device):
  
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
  base_images = multiple_copies.clone()

  base_selection = noise_generator(multiple_copies, pourcentage, reach, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(base_selection, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

  for i in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (10-30) => 40

    # Dont touche the elite

    # Mutate the middle
    middle_mutated = noise_generator(middle_selection, pourcentage, reach, targeted, targeted_channel, device)
    middle_mutated = torch.clamp(middle_mutated, base_images[:elite] - reach, base_images[:elite] + reach)

    # elite + 10 random images
    rand_indices = torch.randperm(batch)[:elite]
    random_selection = base_selection[rand_indices]
    keep_selection = torch.cat((elite_selection, random_selection))
    keep_mutated = noise_generator(keep_selection, pourcentage, reach, targeted, targeted_channel, device)
    keep_mutated = torch.clamp(keep_mutated, base_images[:mid] - reach, base_images[:mid] + reach)
    keep_mutated = noise_generator(keep_mutated, pourcentage, reach, targeted, targeted_channel, device)
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

    print(elite_proba[0], i)

    if elite_proba[0] > targeted_probability:
      return i


  return i