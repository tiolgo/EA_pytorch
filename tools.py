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
import time
import math
import itertools



# TRANSFORMS

transformTensor = transforms.ToTensor()

transformPIL = transforms.ToPILImage()

transformResize = transforms.Resize((224, 224))

transformResizeFigma = transforms.Resize((32, 32))

transformIn = transforms.Compose([
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








# COMPLETE VERSION

# This code work well but is really heavy for some reasons because of torch.randperm() and the large size of pixels
# I might make it lighter if i really need it in the future

def noise_generator(multiple_copies, pourcentage, reach, targeted, targeted_channel, device): # reach = 0.1  ->  -0.1 <= x <= 0.1
                                                                                       # RGB -> 0-R; 1-G; 2-B
    batch_size, channels, height, width = multiple_copies.shape

    # It goes from 0 to- 50176; and to from 0 to 224
    # We do that to retrieve coordinate with // and % rather than using 2 times torch.randperm (pixel_x, pixel_y)

    multiple_copies = multiple_copies.clone().to(device)

    # Change only the blue channel and 90% of its pixels
    if targeted:

      total_pixels = height * width
      pixels_to_change = int(pourcentage * (total_pixels))

      pixel_x_y = torch.randperm(total_pixels)[:pixels_to_change].to(device)  # Shape -> (pixels_to_change)
      pixel_x = pixel_x_y // width
      pixel_y = pixel_x_y % width

      for b in range(batch_size):
        values_to_change = torch.empty(pixels_to_change).uniform_(-reach, reach).to(device)
        multiple_copies[b, targeted_channel, pixel_x, pixel_y] += values_to_change

    # Change every channels of 90% of the pixels
    else:

      total_pixels = channels * height * width
      pixels_to_change = int(pourcentage * (total_pixels))
      pixel_c_x_y = torch.randperm(total_pixels)[:pixels_to_change].to(device)

      pixel_c = pixel_c_x_y // (height * width)
      remainder = pixel_c_x_y % (height * width)
      pixel_x = remainder // width
      pixel_y = remainder % width

      for b in range(batch_size):
        values_to_change = torch.empty(pixels_to_change).uniform_(-reach, reach).to(device)
        multiple_copies[b, pixel_c, pixel_x, pixel_y] += values_to_change

    return multiple_copies.clamp_(0, 1)


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



def find_best_patch_RGB(multiple_copies, divider, elite_matrices, targeted, targeted_channel, device): # reach = 0.1  ->  -0.1 <= x <= 0.1

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

    coord_list = []

    if targeted: channels = 1

    for c in range(channels): # I take each channels to evaluate them individually

      if targeted: c = targeted_channel # Rather than having x matrices changed per channel we have x*3 matrices changed for the targeted channel

      sub_matrices = []
      sub_scores = []

      for h in range(divider): # height iteration
          for w in range(divider): # width iteration

              sub_matrix = image[c, h*sub_matrix_size:(h+1)*sub_matrix_size, w*sub_matrix_size:(w+1)*sub_matrix_size]
              sub_matrices.append(sub_matrix)

              score_horizontal = apply_convolution(sub_matrix, sobel_horizontal).abs().sum().item()
              score_vertical = apply_convolution(sub_matrix, sobel_vertical).abs().sum().item()
              score_diagonal1 = apply_convolution(sub_matrix, sobel_diagonal1).abs().sum().item()
              score_diagonal2 = apply_convolution(sub_matrix, sobel_diagonal2).abs().sum().item()

              score_total = score_horizontal + score_vertical + score_diagonal1 + score_diagonal2

              sub_scores.append(score_total)

      top_indices = torch.topk(torch.tensor(sub_scores, device=device), elite_matrices).indices

      for index in top_indices:
          zone_h = index // divider
          zone_w = index % divider

          for i in range(sub_matrix_size):
              for j in range(sub_matrix_size):
                  x = zone_h * sub_matrix_size + i
                  y = zone_w * sub_matrix_size + j
                  coord_list.append((c, x, y))

    coord_tensor = torch.tensor(coord_list, dtype=torch.long, device=device)
    return coord_tensor


def find_best_patch(multiple_copies, divider, elite_matrices, targeted, targeted_channel, device): # reach = 0.1  ->  -0.1 <= x <= 0.1

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

    coord_list = []
    sub_scores = []

    if targeted: channels = 1

    for c in range(channels): # I take each channels to evaluate them individually

      if targeted: c = targeted_channel # Rather than having x matrices changed per channel we have x*3 matrices changed for the targeted channel

      sub_matrices = []

      for h in range(divider): # height iteration
          for w in range(divider): # width iteration

              sub_matrix = image[c, h*sub_matrix_size:(h+1)*sub_matrix_size, w*sub_matrix_size:(w+1)*sub_matrix_size]
              sub_matrices.append(sub_matrix)

              score_horizontal = apply_convolution(sub_matrix, sobel_horizontal).abs().sum().item()
              score_vertical = apply_convolution(sub_matrix, sobel_vertical).abs().sum().item()
              score_diagonal1 = apply_convolution(sub_matrix, sobel_diagonal1).abs().sum().item()
              score_diagonal2 = apply_convolution(sub_matrix, sobel_diagonal2).abs().sum().item()

              score_total = score_horizontal + score_vertical + score_diagonal1 + score_diagonal2

              sub_scores.append(score_total)

    top_indices = torch.topk(torch.tensor(sub_scores, device=device), (elite_matrices * 3)).indices

    for index in top_indices:
        c = index // (divider * divider)
        relative_index = index % (divider * divider)
        zone_h = relative_index // divider
        zone_w = relative_index % divider

        for i in range(sub_matrix_size):
            for j in range(sub_matrix_size):
                x = zone_h * sub_matrix_size + i
                y = zone_w * sub_matrix_size + j
                coord_list.append((c, x, y))

    coord_tensor = torch.tensor(coord_list, dtype=torch.long, device=device)
    return coord_tensor


def find_random_patch(multiple_copies, divider, elite_matrices, targeted, targeted_channel, device): # reach = 0.1  ->  -0.1 <= x <= 0.1

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

    coord_list = []
    sub_scores = []

    if targeted: channels = 1

    for c in range(channels): # I take each channels to evaluate them individually

      if targeted: c = targeted_channel # Rather than having x matrices changed per channel we have x*3 matrices changed for the targeted channel

      sub_matrices = []

      for h in range(divider): # height iteration
          for w in range(divider): # width iteration

              sub_matrix = image[c, h*sub_matrix_size:(h+1)*sub_matrix_size, w*sub_matrix_size:(w+1)*sub_matrix_size]
              sub_matrices.append(sub_matrix)

              score_horizontal = apply_convolution(sub_matrix, sobel_horizontal).abs().sum().item()
              score_vertical = apply_convolution(sub_matrix, sobel_vertical).abs().sum().item()
              score_diagonal1 = apply_convolution(sub_matrix, sobel_diagonal1).abs().sum().item()
              score_diagonal2 = apply_convolution(sub_matrix, sobel_diagonal2).abs().sum().item()

              score_total = score_horizontal + score_vertical + score_diagonal1 + score_diagonal2

              sub_scores.append(score_total)

    top_indices = torch.randperm(len(sub_scores), device=device)[:elite_matrices * 3]

    for index in top_indices:
        c = index // (divider * divider)
        relative_index = index % (divider * divider)
        zone_h = relative_index // divider
        zone_w = relative_index % divider

        for i in range(sub_matrix_size):
            for j in range(sub_matrix_size):
                x = zone_h * sub_matrix_size + i
                y = zone_w * sub_matrix_size + j
                coord_list.append((c, x, y))

    coord_tensor = torch.tensor(coord_list, dtype=torch.long, device=device)
    return coord_tensor



def noise_generator_patch(multiple_copies, coord_tensor, pourcentage, reach, device):
    batch_size, _, _, _ = multiple_copies.shape
    multiple_copies = multiple_copies.clone().to(device)

    total_coords = coord_tensor.shape[0]
    num_pixels_to_change = int(total_coords * pourcentage)

    indices = torch.randperm(total_coords, device=device)[:num_pixels_to_change]
    sampled_coords = coord_tensor[indices]
    c = sampled_coords[:, 0]
    x = sampled_coords[:, 1]
    y = sampled_coords[:, 2]

    noise = torch.empty((batch_size, num_pixels_to_change), device=device).uniform_(-reach, reach)

    for b in range(batch_size):
        multiple_copies[b].index_put_((c, x, y), noise[b], accumulate=True)

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
    which_channel = torch.randint(0, channels, ()).to(device)
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

def through_model_old(multiple_copies, model, device):
    
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

def through_model(multiple_copies, model, device):
    
    multiple_copies = multiple_copies.clone()

    # multiple_copies.shape -> (40, 3, 225, 224)
    batch_size = multiple_copies.shape[0]

    # One pic after the other, transformer don't handle 4D
    input_copies = torch.stack([transformIn(multiple_copies[i]).to(device) for i in range(batch_size)])

    with torch.no_grad():  # CNN inputs
        logits = model(input_copies)

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
   


def from_rgb_to_ycbcr_batch(tensor_images, device):
    B, C, H, W = tensor_images.shape

    # Matrice de conversion RGB -> YCbCr
    transform_matrix = torch.tensor([
        [0.299,     0.587,     0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5,      -0.418688, -0.081312]
    ], device=device)

    # Mise en forme pour @ produit matriciel
    images_flat = tensor_images.permute(0, 2, 3, 1).reshape(-1, 3)

    # Application de la transformation
    ycbcr_flat = images_flat @ transform_matrix.T

    # Ajout du décalage sur Cb et Cr
    ycbcr_flat[:, 1:] += 0.5

    # Reformater
    ycbcr = ycbcr_flat.view(B, H, W, 3).permute(0, 3, 1, 2)
    ycbcr = torch.clamp(ycbcr, 0.0, 1.0)

    return ycbcr


def from_ycbcr_to_rgb_batch(tensor_images_ycbcr, device):
    B, C, H, W = tensor_images_ycbcr.shape

    # Matrice inverse de conversion YCbCr -> RGB
    inverse_transform_matrix = torch.tensor([
        [1.0,  0.0,      1.402],
        [1.0, -0.344136, -0.714136],
        [1.0,  1.772,    0.0]
    ], device=device)

    # Mise en forme
    images_flat = tensor_images_ycbcr.permute(0, 2, 3, 1).reshape(-1, 3)

    # Décalage des Cb/Cr (recentrage autour de 0)
    images_flat[:, 1:] -= 0.5

    # Application de la transformation inverse
    rgb_flat = images_flat @ inverse_transform_matrix.T

    # Reformater au format batch image
    rgb_images = rgb_flat.view(B, H, W, 3).permute(0, 3, 1, 2)
    rgb_images = torch.clamp(rgb_images, 0.0, 1.0)

    return rgb_images



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


def individual_pixel(tensor_image, channel, height, width, wanted_class, model, device):
   

  image_kill = tensor_image.clone()
  image_push = tensor_image.clone()

  # Avant de commencer il ne faut pas oublier qu'il y a 3 channels et j'ai bien dire
  # qu'il n'y a pas 224x224 pixels mais plutot 224x224x3 pixels

  # On va commencer par modifier le premier premier pixel du premier channel
  image_kill[channel, height, width] = 0
  image_push[channel, height, width] = 1

  batch_image = torch.stack([image_kill, image_push])

  probabilities = through_model(batch_image, model, device)

  wanted_probabilities = probabilities[:, wanted_class]

  if wanted_probabilities[0] >= wanted_probabilities[1]:
    return (wanted_probabilities[0], -1, channel, height, width)
  else:
    return (wanted_probabilities[1], 1, channel, height, width)


def best_pixels(tensor_image, random, pourcentage, wanted_class, model, device): # ce pourcentage rend le programme tres lourd
  # passer 1 revient a passer 100% soit 150528 images dans le CNN

  channels, height, width = tensor_image.shape
  tensor_image = tensor_image.clone()

  if random:
    size = int(pourcentage * (channels * height * width)) # utilise pas itertools car sinon on a pas vrament des pixels random car on target tout une ligne
    random_pixels = set()
    while len(random_pixels) < size:
        pixel = (np.random.randint(0, 3), np.random.randint(0, 224), np.random.randint(0, 224))
        random_pixels.add(pixel)

    random_pixels = list(random_pixels)

    best_pixels_probabilities = [] 

    for pixel in random_pixels:
      proba = individual_pixel(tensor_image, pixel[0], pixel[1], pixel[2], wanted_class, model, device)
      best_pixels_probabilities.append(proba)

    return best_pixels_probabilities
  
  else:  
    channels = [0, 1, 2]
    heights = np.arange(0, 224, 1)
    widths = np.arange(0, 224, 1)

    best_pixels_probabilities = [] 

    for c, h, w in itertools.product(channels, heights, widths):
      proba = individual_pixel(tensor_image, c, h, w, wanted_class, model, device)
      best_pixels_probabilities.append(proba)
    
    return best_pixels_probabilities


def change_pixels(tensor_image, best_pixels_probabilities, pourcentage, reach, device):
  # pourcentage sert a prendre les n meilleur pixels et les modifier, ce n'est pas plus lourd

  channels, height, width = tensor_image.shape
  tensor_image = tensor_image.clone()

  # max pourcentage is 1 which is 100% and represent 3x224x224 => 150528
  # size = int(pourcentage * (channels * height * width))
  size = int(pourcentage * len(best_pixels_probabilities))

  sorted_probabilities = sorted(best_pixels_probabilities, key=lambda x: x[0], reverse=True)
  selection_of_pixels = sorted_probabilities[:size]
  
  for pixel in selection_of_pixels:
      tensor_image[pixel[2], pixel[3], pixel[4]] += (pixel[1] * reach)
  
  tensor_image = torch.clamp(tensor_image, 0.0, 1.0)

  return tensor_image


# YCbCr ne fonctionne pas car il veut qu'on cible des channel pas qu'on dispache le bruit sur plusieurs channels
def individual_pixel_ycbcr(tensor_image, channel, height, width, wanted_class, model, device):
   

  image_kill = from_rgb_to_ycbcr(tensor_image.clone(), device)
  image_push = from_rgb_to_ycbcr(tensor_image.clone(), device)

  # Avant de commencer il ne faut pas oublier qu'il y a 3 channels et j'ai bien dire
  # qu'il n'y a pas 224x224 pixels mais plutot 224x224x3 pixels

  # On va commencer par modifier le premier premier pixel du premier channel
  image_kill[channel, height, width] = 0
  image_push[channel, height, width] = 1

  image_kill = from_rgb_to_ycbcr(image_kill, device)
  image_push = from_rgb_to_ycbcr(image_push, device)

  batch_image = torch.stack([image_kill, image_push])

  probabilities = through_model(batch_image, model, device)

  wanted_probabilities = probabilities[:, wanted_class]

  if wanted_probabilities[0] >= wanted_probabilities[1]:
    return (wanted_probabilities[0], -1, channel, height, width)
  else:
    return (wanted_probabilities[1], 1, channel, height, width)
  


def best_pixels_ycbcr(tensor_image, random, pourcentage, wanted_class, model, device): # ce pourcentage rend le programme tres lourd
  # passer 1 revient a passer 100% soit 150528 images dans le CNN

  channels, height, width = tensor_image.shape
  tensor_image = tensor_image.clone()

  if random:
    size = int(pourcentage * (channels * height * width)) # utilise pas itertools car sinon on a pas vrament des pixels random car on target tout une ligne
    random_pixels = set()
    while len(random_pixels) < size:
        pixel = (np.random.randint(0, 3), np.random.randint(0, 224), np.random.randint(0, 224))
        random_pixels.add(pixel)

    random_pixels = list(random_pixels)

    best_pixels_probabilities = [] 

    for pixel in random_pixels:
      proba = individual_pixel_ycbcr(tensor_image, pixel[0], pixel[1], pixel[2], wanted_class, model, device)
      best_pixels_probabilities.append(proba)

    return best_pixels_probabilities
  
  else:  
    channels = [0, 1, 2]
    heights = np.arange(0, 224, 1)
    widths = np.arange(0, 224, 1)

    best_pixels_probabilities = [] 

    for c, h, w in itertools.product(channels, heights, widths):
      proba = individual_pixel(tensor_image, c, h, w, wanted_class, model, device)
      best_pixels_probabilities.append(proba)
    
    return best_pixels_probabilities
  


def change_pixels_ycbcr(tensor_image, best_pixels_probabilities, pourcentage, reach, device):
  # pourcentage sert a prendre les n meilleur pixels et les modifier, ce n'est pas plus lourd

  channels, height, width = tensor_image.shape
  tensor_image = tensor_image.clone()

  # max pourcentage is 1 which is 100% and represent 3x224x224 => 150528
  # size = int(pourcentage * (channels * height * width))
  size = int(pourcentage * len(best_pixels_probabilities))

  sorted_probabilities = sorted(best_pixels_probabilities, key=lambda x: x[0], reverse=True)
  selection_of_pixels = sorted_probabilities[:size]

  tensor_image = from_rgb_to_ycbcr(tensor_image, device)

  for pixel in selection_of_pixels:
      tensor_image[pixel[2],pixel[3], pixel[4]] += (pixel[1] * reach)


  tensor_image = from_ycbcr_to_rgb(tensor_image, device)

  return tensor_image



def fixed_best_pixels(tensor_image, random_pixels, wanted_class, model, device): # ce pourcentage rend le programme tres lourd
  # passer 1 revient a passer 100% soit 150528 images dans le CNN

  tensor_image = tensor_image.clone()

  best_pixels_probabilities = [] 

  for pixel in random_pixels:
    proba = individual_pixel(tensor_image, pixel[0], pixel[1], pixel[2], wanted_class, model, device)
    best_pixels_probabilities.append(proba)

  return best_pixels_probabilities


def fixed_pick_pixels(tensor_image, pourcentage):
  channels, height, width = tensor_image.shape

  size = int(pourcentage * (channels * height * width)) # utilise pas itertools car sinon on a pas vrament des pixels random car on target tout une ligne

  random_pixels = set()

  while len(random_pixels) < size:
    pixel = (np.random.randint(0, 3), np.random.randint(0, 224), np.random.randint(0, 224))
    random_pixels.add(pixel)

  random_pixels = list(random_pixels)

  return random_pixels