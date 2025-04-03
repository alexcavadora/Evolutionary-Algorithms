import os 
from PIL import Image

# Read image
INPUT_IMAGE = "image_2.png"
image = Image.open(f"{os.path.dirname(__file__)}/{INPUT_IMAGE}")

# Convert to RGBA to handle transparency
if image.mode == 'P':
  image = image.convert('RGBA')

# Transform to black and white
image = image.convert('L')

# Save image
image.save(f"{os.path.dirname(__file__)}/image_bw.png")

# Convert to vector
vector = []
for i in range(image.size[0]):
    for j in range(image.size[1]):
        vector.append(image.getpixel((i, j)))

# Save vector to image