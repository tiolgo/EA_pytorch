
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

def ea_noise_VF(model, enums, image_url, batch, targeted, targeted_channel, wanted_class, height, reach, pourcentage, device):
  
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

  neutral_tensor = torch.full((3, 224, 224), reach, device='cuda')
  batch_neutral_tensor = multiple_copies_generator(neutral_tensor, batch, device)

  base_batch_neutral_tensor = batch_neutral_tensor.clone()

  base_selection = noise_generator(batch_neutral_tensor, pourcentage, reach, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(base_selection, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (10-30) => 40

    # Dont touche the elite

    # Mutate the middle
    middle_mutated = noise_generator(middle_selection, pourcentage, reach, targeted, targeted_channel, device)
    middle_mutated = torch.clamp(middle_mutated, base_batch_neutral_tensor[:elite] - reach, base_batch_neutral_tensor[:elite] + reach)

    # elite + 10 random images
    rand_indices = torch.randperm(batch)[:elite]
    random_selection = base_selection[rand_indices]
    keep_selection = torch.cat((elite_selection, random_selection))
    keep_mutated = noise_generator(keep_selection, pourcentage, reach, targeted, targeted_channel, device)
    keep_mutated = torch.clamp(keep_mutated, base_batch_neutral_tensor[:mid] - reach, base_batch_neutral_tensor[:mid] + reach)
    keep_mutated = noise_generator(keep_mutated, pourcentage, reach, targeted, targeted_channel, device)
    keep_mutated = torch.clamp(keep_mutated, base_batch_neutral_tensor[:mid] - reach, base_batch_neutral_tensor[:mid] + reach)

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

    if targeted:
      combined_images = base_images.clone()
      combined_images[:, targeted_channel, :, :] = base_selection[:, targeted_channel, :, :] + (base_images[:, targeted_channel, :, :] * (1 - 2 * reach))
    else:
      combined_images = base_selection + (base_images * (1 - 2 * reach))

    probabilities_combined = through_model(combined_images, model, device)
    elite_selection_combined, middle_selection_combined, elite_index_combined, elite_proba_combined = selection(combined_images, probabilities_combined, elite, wanted_class, device)
    

    if elite_proba_combined[0] > best_probability:
      best_probability = elite_proba_combined[0]

    
  return best_probability.cpu().item()


def ea_noise_mask_only(model, enums, batch, targeted, targeted_channel, wanted_class, height, reach, pourcentage, device):
  
  # WITH ADDITIONAL NOISE -> ADS UP

  # Free GPU cache
  torch.cuda.empty_cache()

  elite = int(batch/4)
  mid = int(batch/2)

  # 🧑‍🎨 LOAD AND PROCESS THE BASE IMAGE
  best_probability = 0

  neutral_tensor = torch.full((3, 224, 224), reach, device='cuda')
  batch_neutral_tensor = multiple_copies_generator(neutral_tensor, batch, device)

  base_batch_neutral_tensor = batch_neutral_tensor.clone()

  base_selection = noise_generator(batch_neutral_tensor, pourcentage, reach, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(base_selection, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (10-30) => 40

    # Dont touche the elite

    # Mutate the middle
    middle_mutated = noise_generator(middle_selection, pourcentage, reach, targeted, targeted_channel, device)
    middle_mutated = torch.clamp(middle_mutated, base_batch_neutral_tensor[:elite] - reach, base_batch_neutral_tensor[:elite] + reach)

    # elite + 10 random images
    rand_indices = torch.randperm(batch)[:elite]
    random_selection = base_selection[rand_indices]
    keep_selection = torch.cat((elite_selection, random_selection))
    keep_mutated = noise_generator(keep_selection, pourcentage, reach, targeted, targeted_channel, device)
    keep_mutated = torch.clamp(keep_mutated, base_batch_neutral_tensor[:mid] - reach, base_batch_neutral_tensor[:mid] + reach)
    keep_mutated = noise_generator(keep_mutated, pourcentage, reach, targeted, targeted_channel, device)
    keep_mutated = torch.clamp(keep_mutated, base_batch_neutral_tensor[:mid] - reach, base_batch_neutral_tensor[:mid] + reach)

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
      best_image = elite_selection[0]

  
  # image_restored = transformPIL(best_image)

  # plt.imshow(image_restored)
  # plt.axis("off")
  # plt.show()

  return best_probability.cpu().item()





def ea_noise_image(model, enums, batch, targeted, targeted_channel, wanted_class, height, reach, pourcentage, targeted_probability, device):
  
  # WITH ADDITIONAL NOISE -> ADS UP

  # Free GPU cache
  torch.cuda.empty_cache()

  elite = int(batch/4)
  mid = int(batch/2)

  # 🧑‍🎨 LOAD AND PROCESS THE BASE IMAGE

  neutral_tensor = torch.full((3, 224, 224), reach, device='cuda')
  batch_neutral_tensor = multiple_copies_generator(neutral_tensor, batch, device)

  base_batch_neutral_tensor = batch_neutral_tensor.clone()

  base_selection = noise_generator(batch_neutral_tensor, pourcentage, reach, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(base_selection, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(base_selection, probabilities, elite, wanted_class, device)

  for i in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (10-30) => 40

    # Dont touche the elite

    # Mutate the middle
    middle_mutated = noise_generator(middle_selection, pourcentage, reach, targeted, targeted_channel, device)
    middle_mutated = torch.clamp(middle_mutated, base_batch_neutral_tensor[:elite] - reach, base_batch_neutral_tensor[:elite] + reach)

    # elite + 10 random images
    rand_indices = torch.randperm(batch)[:elite]
    random_selection = base_selection[rand_indices]
    keep_selection = torch.cat((elite_selection, random_selection))
    keep_mutated = noise_generator(keep_selection, pourcentage, reach, targeted, targeted_channel, device)
    keep_mutated = torch.clamp(keep_mutated, base_batch_neutral_tensor[:mid] - reach, base_batch_neutral_tensor[:mid] + reach)
    keep_mutated = noise_generator(keep_mutated, pourcentage, reach, targeted, targeted_channel, device)
    keep_mutated = torch.clamp(keep_mutated, base_batch_neutral_tensor[:mid] - reach, base_batch_neutral_tensor[:mid] + reach)

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

    if elite_proba[0] > targeted_probability:
      return elite_selection[0], elite_proba[0]


  return elite_selection[0], elite_proba[0]
