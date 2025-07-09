from PIL import Image
import sys

# Open the image, convert to grayscale, resize to 28x28
img = Image.open(sys.argv[1]).convert('L').resize((28, 28))

# Get pixel data
pixels = img.getdata()

# Convert pixels to comma-separated string
line = ','.join(str(p) for p in pixels)

# Write to file without BOM
with open('output.csv', 'w', encoding='utf-8') as f:
    f.write(line)