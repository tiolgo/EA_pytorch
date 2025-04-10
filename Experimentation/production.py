"""# **IMPORTS**"""

from initialisation import *

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


"""# **MAIN CODE**"""

# EDGE VERSION WITH noise_generator_edge()

# WITH ADDITIONAL NOISE -> ADS UP
# ⚙️ HYPERPARAMETERS

# Free GPU cache
torch.cuda.empty_cache()

ENUMS = 10000

BATCH = 80
ELITES = int(BATCH/4)

BLURRY = True
BLURRINESS = 4 # Needs to be bigger than 224/DIVIDER => sub_matrix_size
TARGETED = False
TARGETED_CHANNEL = 0

THRESHOLD = 0.6 # sucess if the image is considered (THRESHOLD*10)% of the wanted class -> 0.6 = 60% of the wanted class

WANTED_CLASS = 0
UNWANTED_CLASS = 1 # not used

# Height of the swap zone for the crossover
# HEIGHT need relatively big to modify multiple sub_matrices
HEIGHT = 0.25 # significantly increse running time; its the height of the crossover square

REACH = 0.05 # value added to each pixel -> -REACH <= x <= REACH
REACH_NOISE = 0.01 # to add really small mutations

MANUAL = False

# MANUAL = True -> You specify how many sub_matrices to modify yourself
# MANUAL = False -> You specify the min pourcentage of the image that you want to modify

if MANUAL:

  DIVIDER = 56 # 1, 2, 4, 7, 8, 14, 16, 28, 32, 56, 112, 224 -> we are working with 224x224, we need divider of 224
  ELITES_MATRICES = 200
  ELITES_MATRICES_NOISE = 200

  # Keep in mind that you are modifying (ELITES_MATRICES / (DIVIDER**2)) * 100
                                        # (4/49) * 100 =  8% of the base image
                                        # And 8/3 = 2.7% in the case where we target 1 channel

else:

  MIN_POURCENTAGE = 0.025
  MIN_POURCENTAGE_NOISE = 0.025
  DIVIDER = 7 # 1, 2, 4, 7, 8, 14, 16, 28, 32, 56, 112, 224 -> we are working with 224x224, we need divider of 224

  # Don't touch those -> no need to as they are going to be computed to match MIN_POURCENTAGE!
  ELITES_MATRICES = 0
  ELITES_MATRICES_NOISE = 0

  while ELITES_MATRICES/DIVIDER**2 < MIN_POURCENTAGE:
    ELITES_MATRICES += 1

  while ELITES_MATRICES_NOISE/DIVIDER**2 < MIN_POURCENTAGE_NOISE:
    ELITES_MATRICES_NOISE += 1

  print(DIVIDER)
  print(ELITES_MATRICES)
  print(ELITES_MATRICES_NOISE)

# 🧑‍🎨 LOAD AND PROCESS THE BASE IMAGE
image = Image.open("/content/sample_data/dog.jpg")

tensor_image = transformTensor(image).to(device)
multiple_copies = multiple_copies_generator(tensor_image, BATCH)

multiple_copies = noise_generator_edge(multiple_copies, reach=REACH, divider = DIVIDER, elite_matrices=ELITES_MATRICES, blurry=BLURRY, blurriness=BLURRINESS, targeted=TARGETED, targeted_channel=TARGETED_CHANNEL)

# 🤖 MODEL AND SELECTION
probabilities = through_model(multiple_copies)
elites_selection, middle_selection, elites_index, elites_proba = selection(multiple_copies, probabilities, ELITES, WANTED_CLASS, UNWANTED_CLASS, wanted=True)

predicted_classes = torch.argmax(probabilities, dim=-1) # return the classes that the model predicted for each image
print(predicted_classes) # Just for the visual

for i in range(ENUMS):
  print(i)

  # 👨‍💻 COMPUTE EACH SPLIT (20-20) => 40

  top_selection = torch.cat((elites_selection, middle_selection)).clone() # no need for clone here i might remove it after checking

  # 20 -> add noise
  noise_selection = noise_generator_edge(top_selection, reach=REACH_NOISE, divider = DIVIDER, elite_matrices=ELITES_MATRICES_NOISE, blurry=BLURRY, blurriness=BLURRINESS, targeted=TARGETED, targeted_channel=TARGETED_CHANNEL) # must be really small -> it adds up with the iterations

  # 20 -> crossover
  parents_index = parent_generator_fixed(top_selection)
  crossed_copies = crossover_generator(top_selection, parents_index, pourcentage_height=HEIGHT)

  # 20 + 20 = 40 index
  multiple_copies = torch.cat((crossed_copies, noise_selection)) # concatenate the splits together

  # ❗️Now we have the final image set for this iteration to pass through the model!

  # 🤖 MODEL AND SELECTION
  probabilities = through_model(multiple_copies)

  probabilities_wanted = probabilities[:, WANTED_CLASS]
  print("Set of probabilities for the wanted class:")
  print(probabilities_wanted)

  # As long as it is getting bigger we are making progress
  print(probabilities_wanted.sum().item())

  elites_selection, middle_selection, elites_index, elites_proba = selection(multiple_copies, probabilities, ELITES, WANTED_CLASS, UNWANTED_CLASS, wanted=True)

  # 😅 DISPLAY ELITES FOR MONITORING
  # display_elites = [transformPIL(img) for img in elites_selection]

  # fig, axes = plt.subplots(2, 5, figsize=(15, 6))

  # for i, ax in enumerate(axes.flat):
  #     ax.imshow(display_elites[i])
  #     ax.axis("off")

  # plt.show()

  # 👍 RESULTS FOUND?/ HOW GOOD ARE THEY?
  success, success_index = checker(elites_index, elites_proba, THRESHOLD)
  if success:
    print("Adversarial image found!")
    adv_image = multiple_copies[success_index]

    # 🥸 DISPLAY BEST IMAGE
    image_restored = transformPIL(adv_image)
    plt.imshow(image_restored)
    plt.axis("off")
    plt.show()

    break