import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from skimage.io import imread
from skimage.color import rgb2gray
from typing import Union, Optional
from scipy import ndimage
from skimage.io import imread


path = "examples/images"

img_list = os.listdir(path)

rng = np.random.default_rng(len(img_list))
random_ints = rng.integers(low=0, high=len(img_list), size=1)

random_seed = random_ints
random_seed = int(random_seed)

##random image entry 177 in db. Change to img_list[random_seed] for tru random generation
random_img = img_list[0]
print(random_img)

os.chdir(path)
# First image in db collection
# random image 177
file = "spongebob caveman meme.jpg"


def load_image(file: str, grey: bool = False) -> np.ndarray:
    """
    Load an image from a specified file path and optionally convert it to grayscale.

    Parameters:
    file (str): The path to the image file.
    grey (bool): If set to True, the image will be converted to grayscale. Default is False.

    Returns:
    np.ndarray: The loaded image, either in color or grayscale.
    """
    # Load the image from the specified file path
    img = cv.imread(file)

    # Check if the image should be converted to grayscale
    if grey:
        # Convert the image to grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Return the modified or original image
    return img


def show_image(img: np.array, label: str) -> None:
    img = load_image(file)
    cv.imshow(label, img)
    if cv.waitKey(0) & 0xFF == 27:
        cv.destroyAllWindows()
    else:
        cv.waitKey(5000)
        cv.destroyAllWindows()


load_image(file)
show_image(file, "Original Image. Press 0")


def visualize_channels(image_path: str) -> Optional[None]:
    """
    Visualizes the blue, green, and red channels of an input image separately.

    This function loads an image from a specified path, splits its color channels,
    creates grayscale-like images where only one channel retains its information,
    and displays these images alongside the original image.

    Parameters:
        image_path (str): The path to the image file to be visualized.

    Returns:
        None: This function displays the images directly and does not return anything.

    Example usage:
        # Make sure to replace "path/to/your/image.jpg" with an actual image path.
        visualize_channels("path/to/your/image.jpg")
    """

    # Load the image
    image = cv.imread(image_path)

    if image is None:
        print(f"Could not load image at {image_path}")
        return

    # Separate the channels
    blue, green, red = cv.split(image)

    # Create images where only one channel retains its information
    blue_image = cv.merge([blue, np.zeros_like(blue), np.zeros_like(blue)])
    green_image = cv.merge([np.zeros_like(green), green, np.zeros_like(green)])
    red_image = cv.merge([np.zeros_like(red), np.zeros_like(red), red])

    # Display the images
    cv.imwrite("Original.jpg", image)
    cv.imwrite("Blue Channel.jpg", blue_image)
    cv.imwrite("Green Channel.jpg", green_image)
    cv.imwrite("Red Channel.jpg", red_image)


visualize_channels("spongebob caveman meme.jpg")


def process_image_and_display(image_path: Union[str, bytes, "os.PathLike[str]"]):
    """
    Processes an image by applying Sobel filters to compute horizontal and vertical gradients,
    and displays these gradients alongside the original image and their magnitude.

    Args:
        image_path (Union[str, bytes, "os.PathLike[str]"]): The path to the input image, which can be
        a string, a bytes object, or an os.PathLike object representing the path.

    Returns:
        None: Displays the processed images in a matplotlib window.

    Example:
        >>> process_image_and_display("path/to/your/image.jpg")
    """
    # Load and convert the image to grayscale
    image = imread(image_path, as_gray=True).astype("int32")

    # Apply Sobel filters
    sobel_h = ndimage.sobel(image, axis=0)  # horizontal gradient
    sobel_v = ndimage.sobel(image, axis=1)  # vertical gradient

    # Magnitude computation and normalization
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    magnitude *= 255.0 / np.max(magnitude)  # normalization

    # Create subplots and display images
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.gray()  # show the filtered result in grayscale

    axs[0, 0].imshow(image)
    axs[0, 1].imshow(sobel_h)
    axs[1, 0].imshow(sobel_v)
    axs[1, 1].imshow(magnitude)

    titles = ["Original", "Horizontal", "Vertical", "Magnitude"]
    for i, ax in enumerate(axs.ravel()):
        ax.set_title(titles[i])
        ax.axis("off")

    plt.show()


process_image_and_display("spongebob caveman meme.jpg")

# Harris Feature Matching
img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0, 0.04)
dst = cv.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.03 * dst.max()] = [0, 0, 255]


cv.imshow("Harris Feature Matching", img)
if cv.waitKey(0) & 0xFF == 27:
    cv.destroyAllWindows()

# Sift Algorithm
img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray, None)
img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow("Sift Algorithm", img)
if cv.waitKey(0) & 0xFF == 27:
    cv.destroyAllWindows()

# Orb Approach


img = cv.imread(file, cv.IMREAD_GRAYSCALE)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img, None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
plt.imshow(img2), plt.show()

# HOG Approach using SciKit Learn

#image = cv.imread(file, cv.IMREAD_COLOR)
image = cv.imread("spongebob mocking meme.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

fd, hog_image = hog(
    gray,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True,
    channel_axis=None,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis("off")
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title("Input image")


# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis("off")
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title("Histogram of Oriented Gradients")
plt.show()

#FLAN Matcher
MIN_MATCH_COUNT = 10
img1 = cv.imread("spongebob caveman meme.jpg", cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread("spongebob mocking meme.jpg", cv.IMREAD_GRAYSCALE)  # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3, "gray"), plt.show()


##Orb Detector

img1 = cv.imread("spongebob caveman meme.jpg", cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread("spongebob mocking meme.jpg", cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# FLANN parameters
MIN_MATCH_COUNT = 1
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)

des1 = np.float32(des1)
des2 = np.float32(des2)

matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < .25 * n.distance:
        matchesMask[i] = [1, 0]

matches = flann.knnMatch(des1, des2, k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
print(good)
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3, "gray"), plt.show()
cv.imwrite("test.jpg", img3)
