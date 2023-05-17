import cv2
import numpy as np
import os


def stitch_images(img1, img2):
    # Smooth images with Gaussian filter
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB
    orb_params = dict(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=5, firstLevel=0, WTA_K=2,
                      scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=15)
    orb = cv2.ORB_create(**orb_params)

    # Detect ORB features and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match descriptors
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    num_good_matches = int(len(matches) * 0.15)
    matches = matches[:num_good_matches]

    # Draw top matches
    im_matches = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = img2.shape
    im1_reg = cv2.warpPerspective(img1, h, (width, height))

    # Apply cross-dissolve blending
    mask = np.where(im1_reg != 0, 1, 0).astype(np.float32)
    result = (img2 * (1 - mask) + im1_reg * mask).astype(np.uint8)

    # Calculate overlap coordinates
    corners = np.array([[0, 0], [width, 0], [width, height], [
                       0, height]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2), h)[0]
    overlap_coordinates = ' '.join(
        [f"{int(x)} {int(y)}" for x, y in transformed_corners])

    return result, overlap_coordinates


def load_image(path):
    if not os.path.exists(path):
        print(f"Cannot find image at path: {path}")
        return None
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to load image at path: {path}")
    return img


def clean_results_folder():
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def save_overlap_coordinates(file_path, coordinates):
    with open(file_path, 'a') as file:
        file.write(coordinates + '\n')


input1_folder = 'training/input1'
input2_folder = 'training/input2'
output_folder = 'results'
overlap_file = 'overlap.txt'
student_id = '710882199'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Clean the output folder
clean_results_folder()

# Get the list of image names in the input1_folder
image_names = os.listdir(input1_folder)

for image_name in image_names:
    img1_path = os.path.join(input1_folder, image_name)
    img2_path = os.path.join(input2_folder, image_name)
    output_path = os.path.join(output_folder, image_name)

    # load the images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # Check the size of the images and ensure the smaller image is img1
    if img1.shape[0] > img2.shape[0] or img1.shape[1] > img2.shape[1]:
        img1, img2 = img2, img1

    if img1 is not None and img2 is not None:
        # stitch the images together
        stitched_img, overlap_coordinates = stitch_images(img1, img2)

        # If the stitching failed (e.g. not enough matches), skip this image pair
        if stitched_img is None:
            continue

        # save the result
        cv2.imwrite(output_path, stitched_img)

        # Get the coordinates of the overlap area
        overlap_file_path = os.path.join(student_id, overlap_file)

        # Create the student ID folder if it doesn't exist
        os.makedirs(student_id, exist_ok=True)

        # Save the overlap coordinates in the text file
        save_overlap_coordinates(overlap_file_path, overlap_coordinates)
