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
  
def overlay_segments(image, seg_mask):
  """Return different segments predicted by the model overlaid on image."""
  H, W = seg_mask.shape
  image_mask = np.zeros((H, W, 3), dtype=np.uint8)
  colors = np.array(color_palette())
  # convert to a pytorch tensor if seg_mask is not one already
  seg_mask = seg_mask if torch.is_tensor(seg_mask) else torch.tensor(seg_mask)
  unique_labels = torch.unique(seg_mask)
  # map each segment label to a unique color
  for i, label in enumerate(unique_labels):
    image_mask[seg_mask == label.item(), :] = colors[i]
  image = np.array(image)
  # percentage of original image in the final overlaid image
  img_weight = 0.5 
  # overlay input image and the generated segment mask
  img = img_weight * np.array(image) * 255 + (1 - img_weight) * image_mask
  return img.astype(np.uint8)  

def replace_label(mask, label):
  """Replace the segment masks values with label."""
  mask = np.array(mask)
  mask[mask == 255] = label
  return mask

# Image segmentation using Hugging face Pipeline APi

# load the entire image segmentation pipeline
img_segmentation_pipeline = pipeline('image-segmentation', 
                                     model="nvidia/segformer-b5-finetuned-ade-640-640")
