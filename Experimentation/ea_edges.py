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



# EVOLUTIONARY ALGORITHMS

def ea_edges_v1(model, enums, image_url, batch, threshold, blurry, blurriness, targeted, targeted_channel, wanted_class, height, reach, reach_noise, 
             manual, divider, elite_matrices, elite_matrices_noise, min_pourcentage, min_pourcentage_noise, device):
  
  # WITH ADDITIONAL NOISE -> ADS UP

  # Free GPU cache
  torch.cuda.empty_cache()

  elite = int(batch/4)

  if not manual:

    elite_matrices = 0
    elite_matrices_noise = 0

    while elite_matrices/divider**2 < min_pourcentage:
      elite_matrices += 1

    while elite_matrices_noise/divider**2 < min_pourcentage_noise:
      elite_matrices_noise += 1

    print(f"elite_matrices => {elite_matrices}")
    print(f"elite_matrices_noise => {elite_matrices_noise}")

  # 🧑‍🎨 LOAD AND PROCESS THE BASE IMAGE
  image = Image.open(image_url)
  image = transformResize(image)

  tensor_image = transformTensor(image).to(device)
  multiple_copies = multiple_copies_generator(tensor_image, batch, device)

  multiple_copies = noise_generator_edge(multiple_copies, reach, divider, elite_matrices, blurry, blurriness, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(multiple_copies, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(multiple_copies, probabilities, elite, wanted_class, device)

  for i in range(enums):
    print(i)

    # 👨‍💻 COMPUTE EACH SPLIT (20-20) => 40

    top_selection = torch.cat((elite_selection, middle_selection)).clone() # no need for clone here i might remove it after checking

    # 20 -> add noise
    noise_selection = noise_generator_edge(top_selection, reach_noise, divider, elite_matrices_noise, blurry, blurriness, targeted, targeted_channel, device) # must be really small -> it adds up with the iterations

    # 20 -> crossover
    parents_index = parent_generator_fixed(top_selection, device)
    crossed_copies = crossover_generator(top_selection, parents_index, height, device)

    # 20 + 20 = 40 index
    multiple_copies = torch.cat((crossed_copies, noise_selection)) # concatenate the splits together

    # ❗️Now we have the final image set for this iteration to pass through the model!

    # 🤖 MODEL AND SELECTION
    probabilities = through_model(multiple_copies, model, device)
    probabilities_wanted = probabilities[:, wanted_class]

    print("Set of probabilities for the wanted class:")
    print(probabilities_wanted)

    # As long as it is getting bigger we are making progress
    print(f"Sum => {probabilities_wanted.sum().item()}")

    elite_selection, middle_selection, elite_index, elite_proba = selection(multiple_copies, probabilities, elite, wanted_class, device)

    # 😅 DISPLAY ELITE FOR MONITORING
    # display_elite = [transformPIL(img) for img in elite_selection]

    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(display_elite[i])
    #     ax.axis("off")

    # plt.show()

    # 👍 RESULTS FOUND?/ HOW GOOD ARE THEY?
    success, success_index = checker(elite_index, elite_proba, threshold)
    if success:
      print("Adversarial image found!")
      adv_image = multiple_copies[success_index]

      # 🥸 DISPLAY BEST IMAGE
      image_restored = transformPIL(adv_image)
      plt.imshow(image_restored)
      plt.axis("off")
      plt.show()

      return True
    
  return False


# VERSION 2
# It don't return any images, only numerical data to be able to plot graphs

def ea_edges_v2(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, height, reach, reach_noise, 
             manual, divider, elite_matrices, elite_matrices_noise, min_pourcentage, min_pourcentage_noise, device):
  
  # WITH ADDITIONAL NOISE -> ADS UP

  # Free GPU cache
  torch.cuda.empty_cache()

  elite = int(batch/4)

  if not manual:

    elite_matrices = 0
    elite_matrices_noise = 0

    while elite_matrices/divider**2 < min_pourcentage:
      elite_matrices += 1

    while elite_matrices_noise/divider**2 < min_pourcentage_noise:
      elite_matrices_noise += 1

  # 🧑‍🎨 LOAD AND PROCESS THE BASE IMAGE
  best_probability = 0

  image = Image.open(image_url)
  image = transformResize(image)

  tensor_image = transformTensor(image).to(device)
  multiple_copies = multiple_copies_generator(tensor_image, batch, device)

  multiple_copies = noise_generator_edge(multiple_copies, reach, divider, elite_matrices, blurry, blurriness, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(multiple_copies, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(multiple_copies, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (20-20) => 40

    top_selection = torch.cat((elite_selection, middle_selection))

    # 20 -> add noise
    noise_selection = noise_generator_edge(top_selection, reach_noise, divider, elite_matrices_noise, blurry, blurriness, targeted, targeted_channel, device) # must be really small -> it adds up with the iterations

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

    
  return best_probability



# VERSION 3
# Like the version 2 but no noise added, rather than that we generate new noise with base images to stay between -reach and reach

def ea_edges_v3(model, enums, image_url, batch, blurry, blurriness, targeted, targeted_channel, wanted_class, height, reach, 
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

  multiple_copies = noise_generator_edge(multiple_copies, reach, divider, elite_matrices, blurry, blurriness, targeted, targeted_channel, device)

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(multiple_copies, model, device)
  elite_selection, middle_selection, elite_index, elite_proba = selection(multiple_copies, probabilities, elite, wanted_class, device)

  for _ in range(enums):

    # 👨‍💻 COMPUTE EACH SPLIT (20-20) => 40

    top_selection = torch.cat((elite_selection, middle_selection))

    # 20 -> new noises
    noise_selection = noise_generator_edge(base_images, reach, divider, elite_matrices, blurry, blurriness, targeted, targeted_channel, device)

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