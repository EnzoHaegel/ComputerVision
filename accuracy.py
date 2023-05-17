import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# directories
dir1 = "training/sample_resultes"
dir2 = "results"

# Check if directories exist
if not os.path.isdir(dir1):
    print(f"Directory not found: {dir1}")
    exit()

if not os.path.isdir(dir2):
    print(f"Directory not found: {dir2}")
    exit()

# get list of file names in each directory
files1 = os.listdir(dir1)
files2 = os.listdir(dir2)

same_images_count = 0
total_images = len(files2)

# Check if there are images in the results directory
if total_images == 0:
    print("No images found in the results directory.")
    exit()

# iterate through each file in the results directory
for file2 in tqdm(files2, total=total_images):
    # construct full file path
    full_file1 = os.path.join(dir1, file2)
    full_file2 = os.path.join(dir2, file2)

    # Check if the file exists in the training/sample_results directory
    if os.path.isfile(full_file1):
        # open and convert images to numpy arrays
        img1 = np.array(Image.open(full_file1))
        img2 = np.array(Image.open(full_file2))

        # check if they are the same
        if np.array_equal(img1, img2):
            same_images_count += 1
    else:
        print(f"File {file2} not found in {dir1}")

# calculate percentage of same images
same_images_percentage = (same_images_count / total_images) * 100

print(f"Percentage of same images: {same_images_percentage}%")
