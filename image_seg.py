import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import pipeline, SegformerImageProcessor, SegformerForSemanticSegmentation
import requests
from PIL import Image
import urllib.parse as parse
import os

# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

# a function to load an image
def load_image(image_path):
    """Helper function to load images from their URLs or paths."""
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)

img_path = "https://shorthaircatbreeds.com/wp-content/uploads/2020/06/Urban-cat-crossing-a-road-300x180.jpg"
image = load_image(img_path)
image
# convert PIL Image to pytorch tensors
transform = transforms.ToTensor()
image_tensor = image.convert("RGB")
image_tensor = transform(image_tensor)
def color_palette():
  """Color palette to map each class to its corresponding color."""
  return [[0, 128, 128],
          [255, 170, 0],
          [161, 19, 46],
          [118, 171, 47],
          [255, 255, 0],
          [84, 170, 127],
          [170, 84, 127],
          [33, 138, 200],
          [255, 84, 0]]