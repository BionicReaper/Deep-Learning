from PIL import Image
import numpy as np

# Load image and convert to grayscale ("L" mode)
img = Image.open("28x28.png").convert("L")

# Flatten image to 1D array
flat = np.array(img).flatten()

# Save as one-line CSV
with open("output.csv", "w") as f:
    f.write(",".join(map(str, flat)))