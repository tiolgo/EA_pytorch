# IMPORTS

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



# TRANSFORMS

transformTensor = transforms.ToTensor()

transformPIL = transforms.ToPILImage()

transformIn = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation -> ImageNet
])

transformOut = transforms.Compose([
    transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
])



# METHODS

def multiple_copies_generator(image, batch, device):
    return image.unsqueeze(0).expand(batch, -1, -1, -1).to(device) # Format [batch, 3, x, y]

def pixel_to_change_generator(image, lower_pourcentage, upper_pourcentage, device):
    _, height, width = image.shape
    random_pourcentage = torch.FloatTensor(1).uniform_(lower_pourcentage, upper_pourcentage).to(device)
    pixel_to_change = (random_pourcentage * (height * width)).floor().int().item()

    return pixel_to_change#, random_pourcentage.item()

def apply_convolution(tensor, kernel):
    return F.conv2d(tensor.unsqueeze(0).unsqueeze(0), kernel, padding=kernel.shape[2]//2)

# BEST VERSION

# Rather to modify random pixels
# It modifies pixels next to edges to have a greater impact
# To do that i divide the image into x matrices
# And i use sobel convolution matrices to evaluate the likelyhood to have a sharp edge
# I then take x matrices with the most score and randomly change them

# If elite_matrices = 2; it means 2 matrices changed per channel so 6 modification -> 2 x 3
# But if elite_matrices = 2 and targeted is True it means only 2 modification

# ⚠️ Bigger dividers means adding noise closer to the edges!
#     The max amount of elite_matrices is always divider x divider; divider = 7; 7x7 => 49 => max amount of elite_matrices
#     Control the total noise added with divider and elite_matrice

def noise_generator_edge(multiple_copies, reach, divider, elite_matrices, blurry, blurriness, targeted, targeted_channel, device): # reach = 0.1  ->  -0.1 <= x <= 0.1

    batch_size, channels, height, width = multiple_copies.shape

    multiple_copies = multiple_copies.clone().to(device)

    image = multiple_copies[0]

    # Convolution matrices
    sobel_horizontal = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    sobel_vertical = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    sobel_diagonal1 = torch.tensor([[0, 1, 2],
                                    [-1, 0, 1],
                                    [-2, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    sobel_diagonal2 = torch.tensor([[2, 1, 0],
                                    [1, 0, -1],
                                    [0, -1, -2]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)



    sub_matrix_size = int(height/divider) # exemple -> 224/7 => 32; 7x7 => 49 matrices of 32x32 pixels in the 224x224 image

    if targeted: channels = 1

    for c in range(channels): # I take each channels to evaluate them individually

      if targeted: c = targeted_channel # Rather than having x matrices changed per channel we have x*3 matrices changed for the targeted channel

      sub_matrices = []
      sub_scores = []

      for h in range(divider): # height iteration
          for w in range(divider): # width iteration

              sub_matrix = image[c, h*sub_matrix_size:(h+1)*sub_matrix_size, w*sub_matrix_size:(w+1)*sub_matrix_size]
              sub_matrices.append(sub_matrix)

              score_horizontal = apply_convolution(sub_matrix, sobel_horizontal).abs().sum().item() # return a matrice of likelihood to have an edge; we then sum up those likelihood to have a score
              score_vertical = apply_convolution(sub_matrix, sobel_vertical).abs().sum().item()
              score_diagonal1 = apply_convolution(sub_matrix, sobel_diagonal1).abs().sum().item()
              score_diagonal2 = apply_convolution(sub_matrix, sobel_diagonal2).abs().sum().item()

              score_total = score_horizontal + score_vertical + score_diagonal1 + score_diagonal2

              sub_scores.append(score_total)

      top_indices = torch.topk(torch.tensor(sub_scores), elite_matrices).indices # Take the x best score

      for index in top_indices:
        # Set the location for the pixels to change and for each dimensions -> int64 for indexing

        zone_height = int(index//divider) # Get the height start
        zone_width = int(index%divider) # Get the width start

        # Modify the entirety of the matrix

        if blurry:
          # Create a tensor with the values to add (+ and -) Ex: 32x32
          noise = torch.empty(sub_matrix_size, sub_matrix_size, device=device).uniform_(-reach, reach)

          noise = noise.view(1, 1, sub_matrix_size, sub_matrix_size) # interpolate wants batch, channel, height, width -> format

          lowres_noise = F.interpolate(noise, size=(sub_matrix_size // blurriness, sub_matrix_size // blurriness), mode='bicubic') # Can use: nearest; bilinear; bicubic
          smoothed_noise = F.interpolate(lowres_noise, size=(sub_matrix_size, sub_matrix_size), mode='bicubic').squeeze()

          multiple_copies[:, c, zone_height * sub_matrix_size:(zone_height + 1) * sub_matrix_size, zone_width * sub_matrix_size:(zone_width + 1) * sub_matrix_size] += smoothed_noise

        else:
          # Create a tensor with the values to add (+ and -) Ex: 40x32x32
          values_to_change = torch.empty(batch_size, sub_matrix_size, sub_matrix_size).uniform_(-reach, reach).to(device)

          # Add the 40x32x32 to a 40x1x32x32 hot zone of the image
          multiple_copies[:, c, zone_height * sub_matrix_size:(zone_height + 1) * sub_matrix_size, zone_width * sub_matrix_size:(zone_width + 1) * sub_matrix_size] += values_to_change

    return multiple_copies.clamp_(0, 1)

def parent_generator_fixed(multiple_copies, device):
    batch_size = multiple_copies.shape[0]
    parents_index = torch.randperm(batch_size, device=device) # shuffle the indexes

    return parents_index

def crossover_generator(multiple_copies, parents_index, pourcentage_height, device):
  multiple_copies_cross = multiple_copies.clone().to(device)

  batch_size, channels, height, width = multiple_copies.shape

  for i in range(0, len(parents_index), 2):
    parent_index_1 = parents_index[i]
    parent_index_2 = parents_index[i + 1]

    max_size_square = int(multiple_copies.shape[3] * pourcentage_height)  # % hauteur de l'image -> multiple_copies[0].shape[2] prend le nombre de pixel d'une image

    # Swaping zone setting
    which_channel = torch.randint(0, channels, (1,)).to(device)
    size_height = torch.randint(0, max_size_square + 1, (1,)).to(device)
    start_height = torch.randint(0, height - size_height + 1, (1,)).to(device)
    size_width = torch.randint(0, max_size_square + 1, (1,)).to(device)
    start_width = torch.randint(0, width - size_width + 1, (1,)).to(device)

    # Swaping
    swap_zone = multiple_copies_cross[parent_index_1, which_channel, start_height:start_height + size_height, start_width:start_width + size_width,].clone()

    multiple_copies_cross[parent_index_1, which_channel, start_height:start_height + size_height, start_width:start_width + size_width] = \
        multiple_copies_cross[parent_index_2, which_channel, start_height:start_height + size_height, start_width:start_width + size_width]

    multiple_copies_cross[parent_index_2, which_channel, start_height:start_height + size_height, start_width:start_width + size_width] = swap_zone

  return multiple_copies_cross

def through_model(multiple_copies, model, device):
    
    multiple_copies = multiple_copies.clone()

    # multiple_copies.shape -> (40, 3, 225, 224)
    batch_size = multiple_copies.shape[0]

    # One pic after the other, transformer don't handle 4D
    input_copies = torch.stack([transformIn(multiple_copies[i]).to(device) for i in range(batch_size)])

    with torch.no_grad():  # CNN inputs
        outputs = model(input_copies)

    logits = outputs.logits  # logits -> [40, num_classes]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    return probabilities

def selection(multiple_copies, probabilities, elite, wanted_class, device):

    batch_size, channels, height, width = multiple_copies.shape

    probabilities_class = probabilities[:, wanted_class]
    probabilities_class = probabilities_class.to(device)

    sorted_probabilities, indices = torch.sort(probabilities_class, descending=True)
    sorted_probabilities = sorted_probabilities.to(device)
    indices = indices.to(device)

    elite_index = indices[:elite]
    middle_index = indices[elite:batch_size//2]
    low_index = indices[batch_size//2:]

    elite_proba = sorted_probabilities[:elite]
    middle_proba = sorted_probabilities[elite:batch_size//2]
    low_proba = sorted_probabilities[batch_size//2:]

    elite_proba = elite_proba.float()
    middle_proba = middle_proba.float()
    low_proba = low_proba.float()

    elite_selection = multiple_copies[elite_index].to(device)
    middle_selection = multiple_copies[middle_index].to(device)
    low_selection = multiple_copies[low_index].to(device)

    #return elite_selection, middle_selection, low_selection, elite_index, elite_proba
    return elite_selection, middle_selection, elite_index, elite_proba

def from_rgb_to_ycbcr(tensor_image, device):
    transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                     [-0.169, -0.331, 0.499],
                                     [0.499, -0.460, -0.039]], device=device)

    tensor_image_ycbcr = torch.matmul(transform_matrix, tensor_image.view(3, -1)).to(device)

    # Reshape to have the original pytorch shape for images
    tensor_image_ycbcr = tensor_image_ycbcr.view(3, tensor_image.shape[1], tensor_image.shape[2])

    return tensor_image_ycbcr

def checker(elite_index, elite_proba, threshold):

    # Search uniquely in elite
    for i in range(len(elite_proba)):
      if elite_proba[i] >= threshold:
        return True, elite_index[i]
      else:
        return False, -1

# Keep the adversarial image at the same size as the original image in order for the attack to be unnoticeable
# The logic is that the pixels are already pushing the right neurones in order to have wanted probabilities
# By diminushing the weight of every inputs, the probabilities shouldn't move too much

def unnoticeable(adv_image, tensor_image, precision=0.0001): # Max size of an image is 3 x 224 x 224 x 1 = 6 021 120
                                                            # by having 0.0001 as precision we are moving by steps of 602.112
  adv_image = adv_image.clone()

  base_size = tensor_image.sum()
  adv_size = adv_image.sum()
  deficit = torch.ones(3, 224, 224) * precision

  while adv_size > base_size:

    adv_image -= deficit
    adv_size = adv_image.sum()

  return adv_image