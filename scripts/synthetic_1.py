import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import time
from tqdm import tqdm
from datetime import datetime
import yaml
import xml.etree.ElementTree as ET
from PIL import Image
from typing import Dict, List, Optional, Any
import json
from collections import OrderedDict
import warnings
from colorama import init, Fore, Back, Style
import emoji
import shutil
from pyfiglet import Figlet
import random 


f = Figlet(font='slant')
print(f.renderText('Cut Paste Learn'))

# Hides pytorch warnings regarding Gradient and other cross-depencies, which are pinned in DINOcut
warnings.filterwarnings("ignore")
init()
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description="Generates Synthetic Datasets for Object Detection."
)
# add optional arguments
parser.add_argument(
    "-src",
    "-src_dir",
    type=os.path.abspath,
    help="the source directory containing your data for generation",
)
parser.add_argument(
    "-n",
    "-num",
    type=int,
    help="The number of images to be generated. Images will be generated according to a 80/20/20 Train/Test/Val split",
    default=20,
)
parser.add_argument(
    "-io",
    "-image_overlap",
    type=float,
    help="The degree of overlap randomly interpolated by the generator. Set to 0 if you do not want any overlapping images. The default is set to 0.",
    default=0.0,
)
parser.add_argument(
    "-max_obj",
    "-maximum_objects",
    type=int,
    help="The maxinum number of objects to be placed in the scene by the generator. The default is set to 10.",
    default=10,
)
parser.add_argument(
    "-min",
    type=int,
    help="the minimum size of images produced. The default is 150px.",
    default=500,
)
# this will throw an error is a max size is selected that is larger than the background image size. Fix this.
parser.add_argument(
    "-max",
    type=int,
    help="The maximum size of generated images. The defauly is 800px. An error will occur if a max size is selected that is larger than the background image size.",
    default=800,
)
parser.add_argument(
    "-erase",
    "-erase_existing_dataset",
    type=bool,
    help="Boolean value that deletes all files in dataset directory before generation. Default is false, so that labels and images will not be deleted between runs.",
    default=True,
)
parser.add_argument(
    "-config",
    "-config_file",
    type=os.path.abspath,
    help="The location of the config.yaml file containing your albumentations configurations. If not selected synthetic.py will try to load config.py",
    default="dinocut_config.yaml",
)

parser.add_argument(
    "-format",
    "-label_format",
    type=str,
    help="The annotation style for synthetic data generation. Optoins include voc, coc, and yolo (all lowercase strings). Default is set to yolo.",
    default="coco",
)

args = parser.parse_args()

if args.src:
    PATH_MAIN = args.src
else:
    PATH_MAIN = os.path.abspath("data")
    print(Fore.RED+
        f"\n No source directory given. Main Path set to {PATH_MAIN}. Please use python3 synthetic.py -h to learn more.\n"
    )
if args.config:
    yaml_path = args.config
else:
    yaml_path = "config.yaml"


def obj_list():
    """
    Generates a dictionary that maps folder indices to their respective image and mask file paths.

    This function walks through directories located at the path specified by PATH_MAIN, constructing a
    dictionary where each entry corresponds to a folder and contains details like the folder name, and
    paths to all image and mask files within that folder. The function also adjusts for a specific folder
    (typically at index 1) by removing it from the dictionary, assuming it does not follow the same
    directory structure as the others.

    The function assumes the global variables `args.min` and `args.max` are accessible and contain
    relevant bounds which are included in the dictionary for each folder. It also relies on a global
    PATH_MAIN which should be defined externally and points to the main directory path.

    Returns:
        dict: A dictionary where each key is an integer index and each value is a dictionary containing
        the 'folder' name, a list of 'images' file paths, a list of 'masks' file paths, and the
        'longest_min' and 'longest_max' values pulled from global args.

    Raises:
        FileNotFoundError: If PATH_MAIN does not exist or specified folders are not found.
        IndexError: If there is an attempt to delete a non-existing index from the dictionary.

    Note:
        This function directly modifies the dictionary by deleting the entry at index 1 and assumes
        specific global configurations, which might lead to errors if not properly setup.
    """
    obj_dict = {}
    folders = next(os.walk(PATH_MAIN))[1]
    folder_count = len(folders)
    for f in range(folder_count):
        if f < len(folders):
            obj_dict[f] = {
                "folder": folders[f],
                "longest_min": args.min,
                "longest_max": args.max,
            }
    # delete the "background images" since it does not have the same directory structure
    def delete_folders(data):
        folders_to_delete = ["background", "bg_noise"]
        keys_to_delete = [
            key for key, value in data.items() if value["folder"] in folders_to_delete
        ]
        for key in keys_to_delete:
            del data[key]
        return data

    obj_dict = delete_folders(obj_dict)
    temp_list = list(range(len(obj_dict)))
    obj_dict = dict(zip(temp_list, list(obj_dict.values())))
    for k, _ in obj_dict.items():
        folder_name = obj_dict[k]["folder"]

        files_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, folder_name, "images")))
        files_imgs = [
            os.path.join(PATH_MAIN, folder_name, "images", f) for f in files_imgs
        ]

        files_masks = sorted(os.listdir(os.path.join(PATH_MAIN, folder_name, "masks")))
        files_masks = [
            os.path.join(PATH_MAIN, folder_name, "masks", f) for f in files_masks
        ]
        obj_dict[k]["images"] = files_imgs
        obj_dict[k]["masks"] = files_masks
    return obj_dict

files_bg_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, "background")))
files_bg_imgs = [os.path.join(PATH_MAIN, "background", f) for f in files_bg_imgs]

files_bg_noise_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, "bg_noise", "images")))
files_bg_noise_imgs = [
    os.path.join(PATH_MAIN, "bg_noise", "images", f) for f in files_bg_noise_imgs
]
files_bg_noise_masks = sorted(os.listdir(os.path.join(PATH_MAIN, "bg_noise", "masks")))
files_bg_noise_masks = [
    os.path.join(PATH_MAIN, "bg_noise", "masks", f) for f in files_bg_noise_masks
]

def get_img_and_mask(img_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an image and its corresponding mask from specified file paths, convert both to RGB format, and process the mask into a binary format.

    Parameters:
        img_path (str): The file path to the image.
        mask_path (str): The file path to the mask image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the RGB image and the binary mask as NumPy arrays.

    Raises:
        IOError: If the files specified do not exist or cannot be opened.
        ValueError: If the image files are not in a format that can be converted to RGB.

    Example:
        >>> image, binary_mask = get_img_and_mask("path/to/image.jpg", "path/to/mask.jpg")
        This will load the image and mask from the specified paths, convert them to RGB, and convert the mask to a binary format.
    Citation:
        Function modified from https://github.com/alexppppp/synthetic-dataset-object-detection.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"Unable to open image file: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path)
    if mask is None:
        raise IOError(f"Unable to open mask file: {mask_path}")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    mask_b = (
        mask[:, :, 0] == 0
    )  # This is a boolean mask indicating where the mask is applied
    mask = mask_b.astype(np.uint8)  # Convert boolean mask to binary format

    return img, mask


def resize_img(
    img: np.ndarray, desired_max: int, desired_min: int = None
) -> np.ndarray:
    """
    Resize an image to the specified maximum and optional minimum dimensions, maintaining aspect ratio.

    Parameters:
        img (np.ndarray): The input image as a NumPy array.
        desired_max (_int_): The maximum size for the longest dimension of the image.
        desired_min (Optional[_int_]): The minimum size for the shortest dimension of the image.
            If None, the aspect ratio is maintained.

    Returns:
        np.ndarray: The resized image as a NumPy array.

    Raises:
        ValueError: If any of the input dimensions are not positive integers.

    Example:
        >>> img_resized = resize_img(image_array, 800, 600)
        This would resize `image_array` to have the longest dimension be 800 pixels, and the
        shortest dimension proportionally resized to maintain aspect ratio, unless 600 is provided,
        then it resizes the shortest dimension to 600 pixels directly.
    Citation:
        Function modified from https://github.com/alexppppp/synthetic-dataset-object-detection.
    """
    h, w = img.shape[0], img.shape[1]

    longest, shortest = max(h, w), min(h, w)
    longest_new = desired_max
    if desired_min:
        shortest_new = desired_min
    else:
        shortest_new = int(shortest * (longest_new / longest))

    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new

    transform_resize = A.Compose(
        [
            A.Sequential(
                [A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)], p=1
            )
        ]
    )

    transformed = transform_resize(image=img)
    img_r = transformed["image"]

    return img_r

def load_transformations_from_yaml(yaml_path: str) -> list:
    """
    Load transformation configurations from a YAML file and create corresponding Albumentations objects.

    Parameters:
        yaml_path (str): Path to the YAML file containing the transformation configurations.

    Returns:
        dict: A dictionary containing Albumentations Compose objects for background and object transformations.
    """
    # Load YAML file
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
        #strips away other items from dinocut.yaml to just focus on the image augmentation settings. 
        config = list(config.items())[-2:]
        config = dict(config)

    # Helper function to construct an Albumentations Compose object from a list of transformations
    def construct_transformation(transform_config):
        transformations = []
        for transform in transform_config:
            for key, value in transform.items():
                transform_class = getattr(A, key)
                transformations.append(transform_class(**value))
        return A.Compose(transformations)

    # Create transformation objects from YAML config
    transformations = {
        key: construct_transformation(value) for key, value in config.items()
    }
    print(Fore.BLUE+ "Background and Object Transformations loaded successfully.")
    return transformations


transformation_objects = load_transformations_from_yaml(yaml_path)
transforms_bg_obj = transformation_objects["transforms_bg_obj"]
transforms_obj = transformation_objects["transforms_obj"]

def resize_transform_obj(
    img: np.ndarray,
    mask: np.ndarray,
    longest_min: int,
    longest_max: int,
    transforms: bool = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resizes an image and its corresponding mask to a new size, randomly determined within specified bounds for the longest dimension, and optionally applies additional transformations.

    Parameters:
        img (np.ndarray): The image to be transformed, expected to be a NumPy array.
        mask (np.ndarray): The mask corresponding to the image, expected to be a NumPy array of the same dimensions as the image.
        longest_min (int): The minimum value for the longest dimension of the resized image and mask.
        longest_max (int): The maximum value for the longest dimension of the resized image and mask.
        transforms (bool): A function or composed set of transformations to apply to the resized image and mask. If False, no additional transformations are applied.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the resized and possibly transformed image and mask.

    Raises:
        ValueError: If the `longest_min` is greater than `longest_max`.
        IndexError: If the dimensions of `mask` do not match the dimensions of `img`.

    Example:
        >>> image, mask = resize_transform_obj(image_array, mask_array, 300, 500)
        This will resize `image_array` and `mask_array` to a random size between 300 and 500 pixels for the longest dimension,
        maintaining the aspect ratio, and apply additional transformations if provided.
    Citation:
        Function modified from https://github.com/alexppppp/synthetic-dataset-object-detection.
    """
    h, w = mask.shape[:2]

    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new / longest))

    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new

    transform_resize = A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)

    transformed_resized = transform_resize(image=img, mask=mask)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]

    if transforms:
        transformed = transforms(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"]

    return img_t, mask_t


def add_obj(
    img_comp: np.ndarray,
    mask_comp: np.ndarray,
    img: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    idx: int,
) -> tuple:
    """
    Incorporates an object and its mask into an existing image composition at specified coordinates.

    Parameters:
        img_comp (np.ndarray): The existing composition of images in which the new image will be integrated.
        mask_comp (np.ndarray): The existing composition of masks corresponding to img_comp.
        img (np.ndarray): The image of the object to be added.
        mask (np.ndarray): The binary mask of the object to be added.
        x (int): The x-coordinate where the center of the object image is to be placed.
        y (int): The y-coordinate where the center of the object image is to be placed.
        idx (int): The index used to identify the object in the mask composition.

    Returns:
        tuple: A tuple containing the updated image composition, mask composition, and the segment of the added mask.

    Description:
        The function adjusts the coordinates to ensure the object image is centered at the specified (x, y) location.
        It then calculates the intersection of the object image with the bounds of the image composition, applies the object
        image and mask to the corresponding regions, and updates the mask composition using the provided idx. The function
        handles different cases based on the boundaries and whether the specified coordinates are inside the composition.
    Citation:
        Function modified from https://github.com/alexppppp/synthetic-dataset-object-detection.
    """
    h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]

    h, w = img.shape[0], img.shape[1]

    x = x - int(w / 2)
    y = y - int(h / 2)

    mask_b = mask == 1
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)

    if x >= 0 and y >= 0:
        h_part = h - max(
            0, y + h - h_comp
        )  # h_part - part of the image which gets into the frame of img_comp along y-axis
        w_part = w - max(
            0, x + w - w_comp
        )  # w_part - part of the image which gets into the frame of img_comp along x-axis

        img_comp[y : y + h_part, x : x + w_part, :] = (
            img_comp[y : y + h_part, x : x + w_part, :]
            * ~mask_rgb_b[0:h_part, 0:w_part, :]
            + (img * mask_rgb_b)[0:h_part, 0:w_part, :]
        )
        mask_comp[y : y + h_part, x : x + w_part] = (
            mask_comp[y : y + h_part, x : x + w_part] * ~mask_b[0:h_part, 0:w_part]
            + (idx * mask_b)[0:h_part, 0:w_part]
        )
        mask_added = mask[0:h_part, 0:w_part]

    elif x < 0 and y < 0:
        h_part = h + y
        w_part = w + x

        img_comp[0 : 0 + h_part, 0 : 0 + w_part, :] = (
            img_comp[0 : 0 + h_part, 0 : 0 + w_part, :]
            * ~mask_rgb_b[h - h_part : h, w - w_part : w, :]
            + (img * mask_rgb_b)[h - h_part : h, w - w_part : w, :]
        )
        mask_comp[0 : 0 + h_part, 0 : 0 + w_part] = (
            mask_comp[0 : 0 + h_part, 0 : 0 + w_part]
            * ~mask_b[h - h_part : h, w - w_part : w]
            + (idx * mask_b)[h - h_part : h, w - w_part : w]
        )
        mask_added = mask[h - h_part : h, w - w_part : w]

    elif x < 0 and y >= 0:
        h_part = h - max(0, y + h - h_comp)
        w_part = w + x

        img_comp[y : y + h_part, 0 : 0 + w_part, :] = (
            img_comp[y : y + h_part, 0 : 0 + w_part, :]
            * ~mask_rgb_b[0:h_part, w - w_part : w, :]
            + (img * mask_rgb_b)[0:h_part, w - w_part : w, :]
        )
        mask_comp[y : y + h_part, 0 : 0 + w_part] = (
            mask_comp[y : y + h_part, 0 : 0 + w_part]
            * ~mask_b[0:h_part, w - w_part : w]
            + (idx * mask_b)[0:h_part, w - w_part : w]
        )
        mask_added = mask[0:h_part, w - w_part : w]

    elif x >= 0 and y < 0:
        h_part = h + y
        w_part = w - max(0, x + w - w_comp)

        img_comp[0 : 0 + h_part, x : x + w_part, :] = (
            img_comp[0 : 0 + h_part, x : x + w_part, :]
            * ~mask_rgb_b[h - h_part : h, 0:w_part, :]
            + (img * mask_rgb_b)[h - h_part : h, 0:w_part, :]
        )
        mask_comp[0 : 0 + h_part, x : x + w_part] = (
            mask_comp[0 : 0 + h_part, x : x + w_part]
            * ~mask_b[h - h_part : h, 0:w_part]
            + (idx * mask_b)[h - h_part : h, 0:w_part]
        )
        mask_added = mask[h - h_part : h, 0:w_part]

    return img_comp, mask_comp, mask_added



def create_bg_with_noise(
    files_bg_imgs: List[str],
    files_bg_noise_imgs: List[str],
    files_bg_noise_masks: List[str],
    bg_max: int,
    bg_min: int,
    max_objs_to_add: int,
    longest_bg_noise_max: int,
    longest_bg_noise_min: int,
    blank_bg: bool,
) -> np.ndarray:
    """
    Creates a background image by optionally adding noise objects to either a plain or pre-existing background image.

    Parameters:
        files_bg_imgs (List[str]): File paths to background images.
        files_bg_noise_imgs (List[str]): File paths to noise images that can be added to the background.
        files_bg_noise_masks (List[str]): File paths to masks corresponding to the noise images.
        bg_max (int): The maximum dimension for resizing the background image.
        bg_min (int): The minimum dimension for resizing the background image.
        max_objs_to_add (int): The maximum number of noise objects to add to the background.
        longest_bg_noise_max (int): The maximum dimension for resizing the noise objects.
        longest_bg_noise_min (int): The minimum dimension for resizing the noise objects.
        blank_bg (bool): If True, starts with a blank white background; otherwise, uses a random background image.

    Returns:
        np.ndarray: The generated background image with added noise objects in CV2 RGB format.

    Notes:
        This function randomly selects a background image from `files_bg_imgs` unless a blank background is specified.
        It then randomly selects and adds noise objects up to `max_objs_to_add` times to the background. Each object and its
        mask are resized and transformed before being added. The resizing and transformation of images depend on the specified
        minimum and maximum dimensions. Noise objects are chosen randomly, and their placement on the background is also random.
    Citation:
        Function modified from https://github.com/alexppppp/synthetic-dataset-object-detection.
    """
    if blank_bg:
        img_comp_bg = np.ones((bg_min, bg_max, 3), dtype=np.uint8) * 255
        mask_comp_bg = np.zeros((bg_min, bg_max), dtype=np.uint8)
    else:
        idx = np.random.randint(len(files_bg_imgs))
        img_bg = cv2.imread(files_bg_imgs[idx])
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
        img_comp_bg = resize_img(img_bg, bg_max, bg_min)
        mask_comp_bg = np.zeros(
            (img_comp_bg.shape[0], img_comp_bg.shape[1]), dtype=np.uint8
        )

    for i in range(1, np.random.randint(max_objs_to_add) + 2):
        idx = np.random.randint(len(files_bg_noise_imgs))
        img, mask = get_img_and_mask(
            files_bg_noise_imgs[idx], files_bg_noise_masks[idx]
        )
        x, y = np.random.randint(img_comp_bg.shape[1]), np.random.randint(
            img_comp_bg.shape[0]
        )
        img_t, mask_t = resize_transform_obj(
            img,
            mask,
            longest_bg_noise_min,
            longest_bg_noise_max,
            transforms=transforms_bg_obj,
        )
        img_comp_bg, _, _ = add_obj(img_comp_bg, mask_comp_bg, img_t, mask_t, x, y, i)

    return img_comp_bg


def check_areas(
    mask_comp: np.ndarray, obj_areas: list, overlap_degree: float = args.io
) -> bool:
    """
    Checks if the area overlap between objects in an image composition exceeds a specified degree.

    Parameters:
        mask_comp (np.ndarray): A 2D array representing the mask composition where different values correspond to different objects.
        obj_areas (list): A list containing the area (in pixels) of each object when it was first placed in the composition.
        overlap_degree (float, optional): The maximum allowable proportion of overlap relative to the original object area. Defaults to 0.3.

    Returns:
        bool: True if the area overlaps are within the acceptable limits, False otherwise.

    Description:
        The function compares the areas of each object in the composition mask to their original areas when first placed.
        If any object's current area in the composition is less than its original area minus the allowed overlap degree,
        it suggests significant overlap with other objects and the function returns False. This is crucial for applications
        where object independence in detection or analysis is necessary.
    Citation:
        Function modified from https://github.com/alexppppp/synthetic-dataset-object-detection.
    """
    obj_ids = np.unique(mask_comp).astype(np.uint8)[
        1:-1 #-1
    ]  # Skip the background (typically 0) and exclude the last ID if unused
    #print(obj_ids) debugging
    masks = mask_comp == obj_ids[:, None, None]

    ok = True  # Assume no excessive overlap by default

    # Check for consistency in object IDs, implying no new IDs without corresponding masks
    if len(np.unique(mask_comp)) != np.max(mask_comp) + 1:
        ok = False
        return ok

    # Verify each object's area against its initial area factoring in permissible overlap
    for idx, mask in enumerate(masks):
        if np.count_nonzero(mask) / obj_areas[idx] < 1 - overlap_degree:
            ok = False
            break

    return ok


def create_composition(
    img_comp_bg: np.ndarray,
    max_objs: int,
    overlap_degree: float,
    max_attempts_per_obj: int,
) -> tuple:
    """
    Creates a composite image by randomly placing objects onto a background image with constraints on overlap and number of attempts.

    Parameters:
        img_comp_bg (np.ndarray): The background image on which objects will be added.
        max_objs (int): The maximum number of objects to attempt to add to the background.
        overlap_degree (float): A threshold for the acceptable degree of overlap between objects.
        max_attempts_per_obj (int): The maximum number of attempts to place each object before moving to the next.

    Returns:
        tuple: A tuple containing the composite image (np.ndarray), the composite mask (np.ndarray),
               a list of labels corresponding to the objects added (list), and a list of areas for each object mask (list).

    Description:
        The function iterates over a randomly determined number of objects (not exceeding max_objs) to add to the background.
        Each object is selected from a pre-defined dictionary `obj_dict` which is expected to be generated by the `obj_list` function.
        The objects are resized and transformed based on predefined criteria and are attempted to be added to the image composition,
        checking for overlap constraints. If the placement of an object violates the overlap degree, the attempt is aborted and retried.
        The process respects the maximum number of placement attempts for each object.
    Citation:
        Function modified from https://github.com/alexppppp/synthetic-dataset-object-detection.
    """
    obj_dict = obj_list()
    img_comp = img_comp_bg.copy()
    h, w = img_comp.shape[0], img_comp.shape[1]
    mask_comp = np.zeros((h, w), dtype=np.uint8)

    obj_areas = []
    labels_comp = []
    num_objs = np.random.randint(max_objs) + 2

    i = 1

    for _ in range(1, num_objs):
        obj_idx = np.random.randint(len(obj_dict))   #+ 1

        for _ in range(max_attempts_per_obj):
            imgs_number = len(obj_dict[obj_idx]["images"])
            idx = np.random.randint(imgs_number)
            img_path = obj_dict[obj_idx]["images"][idx]
            mask_path = obj_dict[obj_idx]["masks"][idx]
            img, mask = get_img_and_mask(img_path, mask_path)

            x, y = np.random.randint(w), np.random.randint(h)
            longest_min = obj_dict[obj_idx]["longest_min"]
            longest_max = obj_dict[obj_idx]["longest_max"]
            img, mask = resize_transform_obj(
                img, mask, longest_min, longest_max, transforms=transforms_obj
            )

            if i == 1:
                img_comp, mask_comp, mask_added = add_obj(
                    img_comp, mask_comp, img, mask, x, y, i
                )
                obj_areas.append(np.count_nonzero(mask_added))
                labels_comp.append(obj_idx)
                i += 1
                break
            else:
                img_comp_prev, mask_comp_prev = img_comp.copy(), mask_comp.copy()
                img_comp, mask_comp, mask_added = add_obj(
                    img_comp, mask_comp, img, mask, x, y, i
                )
                ok = check_areas(mask_comp, obj_areas, overlap_degree)
                if ok:
                    obj_areas.append(np.count_nonzero(mask_added))
                    labels_comp.append(obj_idx)
                    i += 1
                    break
                else:
                    img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()

    return img_comp, mask_comp, labels_comp, obj_areas


def create_yolo_annotations(mask_comp, labels_comp):
    comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]
    
    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
    masks = mask_comp == obj_ids[:, None, None]

    # Adjust labels_comp length to match masks length
    if len(masks) != len(labels_comp):
        if len(masks) < len(labels_comp):
            labels_comp = labels_comp[:len(masks)]
        else:
            print(f"Warning: Not enough labels for all objects. Expecting {len(masks)}, but got {len(labels_comp)}.")
            labels_comp.extend([-1] * (len(masks) - len(labels_comp)))  # Adding placeholder labels

    annotations_yolo = []
    for i in range(len(labels_comp)):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        annotations_yolo.append([labels_comp[i],
                                 round(xc/comp_w, 5),
                                 round(yc/comp_h, 5),
                                 round(w/comp_w, 5),
                                 round(h/comp_h, 5)])

    return annotations_yolo

def generate_dataset(imgs_number: int, folder: str, split: str = "train") -> None:
    """
    Generates a dataset of synthetic images and corresponding annotations based on specified parameters.

    Parameters:
        imgs_number (int): The number of images to generate.
        folder (str): The base directory where the dataset should be stored.
        split (str, optional): The category of the dataset, e.g., 'train', 'test', 'val'. Defaults to 'train'.

    Description:
        This function generates a specified number of synthetic images by creating backgrounds with noise
        and adding compositions of objects. Each image is then saved in a specified folder under a specific
        dataset split category. Corresponding annotations in YOLO format are also generated and saved.
        Execution time per image and total time are calculated and printed at the end of the run.

    Effects:
        - Images are saved to '{folder}/{split}/images' directory.
        - Annotations are saved to '{folder}/{split}/labels' directory.
        - Prints the total time taken and time per image to the console.

    Raises:
        IOError: If there are issues writing the files to the disk.
        Exception: If any of the called functions (`create_bg_with_noise`, `create_composition`, etc.) fail.
    Citation:
        Function modified from https://github.com/alexppppp/synthetic-dataset-object-detection.
    """
    time_start = time.time()
    timing = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    
    for j in tqdm(range(imgs_number)):
        while True:
            try:
                img_comp_bg = create_bg_with_noise(
                    files_bg_imgs,
                    files_bg_noise_imgs,
                    files_bg_noise_masks,
                    max_objs_to_add=60,
                    bg_max=1920,
                    bg_min=1080,
                    longest_bg_noise_max=1000,
                    longest_bg_noise_min=200,
                    blank_bg=False,
                )

                img_comp, mask_comp, labels_comp, _ = create_composition(
                    img_comp_bg,
                    max_objs=args.max_obj,
                    overlap_degree=args.io,
                    max_attempts_per_obj=10,
                )

                annotations_yolo = create_yolo_annotations(mask_comp, labels_comp)

                if not annotations_yolo:
                    raise ValueError("No annotations found, regenerating the image...")

                # If there are annotations, save the image and break the loop
                img_comp = cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(folder, split, "images/{}" + timing + ".jpg").format(j),
                    img_comp,
                )

                for i in range(len(annotations_yolo)):
                    with open(
                        os.path.join(folder, split, "labels/{}" + timing + ".txt").format(j),
                        "a",
                    ) as f:
                        f.write(" ".join(str(el) for el in annotations_yolo[i]) + "\n")
                
                break  # Exit the while loop since the image and annotations are saved successfully

            except ValueError as e:
                print(e)
                continue  # Regenerate the image if no annotations are found

    time_end = time.time()
    time_total = round(time_end - time_start)
    time_per_img = round((time_end - time_start) / imgs_number, 1)
    print(Fore.BLUE +
        "Generation of {} synthetic images is completed. It took {} seconds, or {} seconds per image".format(
            imgs_number, time_total, time_per_img
        )
    )
    print(Fore.GREEN + "Images are stored in '{}'".format(os.path.join(folder, split, "images")))
    print(
        Fore.GREEN +"Annotations are stored in '{}'".format(os.path.join(folder, split, "labels"))
    )

def mkdir() -> None:
    """
    Creates directories for dataset storage and manages file system navigation.

    This function performs a series of operations to ensure that the necessary directories for storing
    a dataset are present. It attempts to create a 'dataset' directory, and if it already exists, it changes
    the working directory to it. Subdirectories for 'train', 'test', and 'val' datasets, along with their
    respective 'images' and 'labels' subdirectories, are also created. The function handles file system
    navigation and provides user feedback about the operations performed.

    During the process, it prints the home directory, the main data path (assumed to be specified by
    global `PATH_MAIN`), and the dataset directory. If specified by command-line arguments (using `args.erase`),
    it can optionally delete existing datasets to allow for a clean regeneration.

    Raises:
        OSError: If directory creation or navigation fails, or if file deletion encounters issues.
        AttributeError: If required globals like `PATH_MAIN` or `args` are not set.

    Notes:
        - The function assumes that the global variable `dataset` should be defined outside or it will be set
          dynamically in case of exceptions during directory operations.
        - It relies on the argparse library to handle command-line arguments, specifically checking `args.erase`
          to determine if existing data files should be deleted.
        - The function does not return any values but prints details about the operations and their outcomes
          to the console.
    """
    try:
        print(Fore.BLUE+"\n\n Checking Project Paths:")
        home = os.path.abspath(os.getcwd())
        try:
            os.mkdir(dataset)
            os.chdir(dataset)
        except:
            os.chdir(dataset)
        dir_list = ["train", "test", "val"]
        sub_dir_list = ["images", "labels"]
        for x in dir_list:
            os.mkdir(x)
        print(Fore.BLUE+"\n\t Home Directory: '{}' ".format(home))
        print(Fore.BLUE+"\n\t Data For Generation: '{}'".format(PATH_MAIN))
    except:
        dataset = os.path.join(home, "dataset")
        print(Fore.BLUE+"\n\t Home Directory:'{}' ".format(home))
        print(Fore.BLUE+"\n\t Data For Generation: '{}' ".format(PATH_MAIN))
        print(Fore.BLUE+"\n\t Generated Datasets Stored in: '{}' ".format(dataset))
        # insert if statement for argparse libary that runs clean remove existing dataset -del if set to true. Default will be false..
        total_number = []
        if args.erase == True:
            print(Fore.RED+"\n\tThe Erase function has been enabled \n")
            print(Fore.RED+"\n Removing Old Datasets from: {}".format(dataset))
            for root, dirs, files in os.walk(dataset):
                for name in tqdm(files):
                    total_number.append(files)
                    if name.endswith(".jpg"):
                        selected_files = os.path.join(root, name)
                        os.remove(selected_files)
                    if name.endswith(".xml"):
                        selected_files = os.path.join(root, name)
                        os.remove(selected_files)
                    if name.endswith(".txt"):
                        selected_files = os.path.join(root, name)
                        os.remove(selected_files)
                    if  name.endswith(".json"):
                        selected_files = os.path.join(root, name)
                        os.remove(selected_files)
            print(Fore.RED+"\n{} files were deleted.".format(len(total_number)))
        else:
            for root, dirs, files in os.walk(dataset):
                for name in files:
                    if name.endswith(".jpg"):
                        selected_files = os.path.join(root, name)
                    if name.endswith(".txt" or ".xml" or ".json"):
                        selected_files = os.path.join(root, name)
            print(Fore.RED+"\n\t {} files were deleted.".format(len(total_number)))
            print(
                Fore.BLUE+"\n\t There are {} labels and images in this dataset.".format(
                    len(selected_files)
                )
            )
            print(Fore.GREEN+"\n Beginning Data Generation\n")
            pass
'''
def mkdir() -> None:
    """
    Creates directories for dataset storage and manages file system navigation.

    This function performs a series of operations to ensure that the necessary directories for storing
    a dataset are present. It attempts to create a 'dataset' directory, and if it already exists, it changes
    the working directory to it. Subdirectories for 'train', 'test', and 'val' datasets, along with their
    respective 'images' and 'labels' subdirectories, are also created. The function handles file system
    navigation and provides user feedback about the operations performed.

    During the process, it prints the home directory, the main data path (assumed to be specified by
    global `PATH_MAIN`), and the dataset directory. If specified by command-line arguments (using `args.erase`),
    it can optionally delete existing datasets to allow for a clean regeneration.

    Raises:
        OSError: If directory creation or navigation fails, or if file deletion encounters issues.
        AttributeError: If required globals like `PATH_MAIN` or `args` are not set.

    Notes:
        - The function assumes that the global variable `dataset` should be defined outside or it will be set
          dynamically in case of exceptions during directory operations.
        - It relies on the argparse library to handle command-line arguments, specifically checking `args.erase`
          to determine if existing data files should be deleted.
        - The function does not return any values but prints details about the operations and their outcomes
          to the console.
    """
    try:
        print(Fore.BLUE+"\n\n Checking Project Paths:")
        home = os.path.abspath(os.getcwd())
        try:
            os.mkdir(dataset)
            os.chdir(dataset)
        except:
            os.chdir(dataset)
        dir_list = ["train", "test", "val"]
        sub_dir_list = ["images", "labels"]
        for x in dir_list:
            os.mkdir(x)
        print(Fore.BLUE+"\n\t Home Directory: '{}' ".format(home))
        print(Fore.BLUE+"\n\t Data For Generation: '{}'".format(PATH_MAIN))
    except:
        dataset = os.path.join(home, "dataset")
        print(Fore.BLUE+"\n\t Home Directory:'{}' ".format(home))
        print(Fore.BLUE+"\n\t Data For Generation: '{}' ".format(PATH_MAIN))
        print(Fore.BLUE+"\n\t Generated Datasets Stored in: '{}' ".format(dataset))
        # insert if statement for argparse libary that runs clean remove existing dataset -del if set to true. Default will be false..
        total_number = []
        if args.erase == True:
            print(Fore.RED+"\n\tThe Erase function has been enabled \n")
            print(Fore.RED+"\n Removing Old Datasets from: {}".format(dataset))
            for root, dirs, files in os.walk(dataset):
                for name in tqdm(files):
                    total_number.append(files)
                    if name.endswith(".jpg"):
                        selected_files = os.path.join(root, name)
                        os.remove(selected_files)
                    if name.endswith(".txt"):
                        selected_files = os.path.join(root, name)
                        os.remove(selected_files)
            print(Fore.RED+"\n{} files were deleted.".format(len(total_number)))
        else:
            for root, dirs, files in os.walk(dataset):
                for name in files:
                    if name.endswith(".jpg"):
                        selected_files = os.path.join(root, name)
                    if name.endswith(".txt"):
                        selected_files = os.path.join(root, name)
            print(Fore.RED+"\n\t {} files were deleted.".format(len(total_number)))
            print(
                Fore.BLUE+"\n\t There are {} labels and images in this dataset.".format(
                    len(selected_files)
                )
            )
            print(Fore.GREEN+"\n Beginning Data Generation\n")
            pass

def mkdir() -> None:
    """
    Creates directories for dataset storage and manages file system navigation.

    This function ensures that the necessary directories for storing a dataset are present. It attempts to create a 
    'dataset' directory, and if it already exists, it changes the working directory to it. Subdirectories for 'train', 
    'test', and 'val' datasets, along with their respective 'images' and 'labels' subdirectories, are also created. 
    The function handles file system navigation and provides user feedback about the operations performed.

    During the process, it prints the home directory, the main data path (assumed to be specified by global `PATH_MAIN`), 
    and the dataset directory. If specified by command-line arguments (using `args.erase`), it can optionally delete 
    existing datasets to allow for a clean regeneration.

    Raises:
        OSError: If directory creation or navigation fails, or if file deletion encounters issues.
        AttributeError: If required globals like `PATH_MAIN` or `args` are not set.

    Notes:
        - The function assumes that the global variable `dataset` should be defined outside or it will be set
          dynamically in case of exceptions during directory operations.
        - It relies on the argparse library to handle command-line arguments, specifically checking `args.erase`
          to determine if existing data files should be deleted.
        - The function does not return any values but prints details about the operations and their outcomes
          to the console.
    """
    try:
        print("\n\nChecking Project Paths:")
        home = os.path.abspath(os.getcwd())
        dataset = os.path.join(home, "dataset")
        
        # Ensure the dataset directory exists
        if not os.path.exists(dataset):
            os.mkdir(dataset)
        os.chdir(dataset)

        dir_list = ["train", "test", "val"]
        sub_dir_list = ["images", "labels"]

        for main_dir in dir_list:
            main_dir_path = os.path.join(dataset, main_dir)
            if not os.path.exists(main_dir_path):
                os.mkdir(main_dir_path)
            for sub_dir in sub_dir_list:
                sub_dir_path = os.path.join(main_dir_path, sub_dir)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(sub_dir_path)
        
        print("\n\tHome Directory: '{}'".format(home))
        print("\n\tData For Generation: '{}'".format(PATH_MAIN))

        # Handle erasure of existing datasets if specified
        if args.erase:
            print("\n\tThe Erase function has been enabled")
            print("\nRemoving Old Datasets from: {}".format(dataset))
            total_number = 0
            for root, dirs, files in os.walk(dataset):
                for name in tqdm(files):
                    file_path = os.path.join(root, name)
                    if name.endswith(".jpg") or name.endswith(".txt"):
                        os.remove(file_path)
                        total_number += 1
            print("\n{} files were deleted.".format(total_number))
        else:
            total_files = 0
            for root, dirs, files in os.walk(dataset):
                total_files += len([name for name in files if name.endswith(".jpg") or name.endswith(".txt")])
            print("\n\tThere are {} labels and images in this dataset.".format(total_files))
            print("\nBeginning Data Generation\n")

    except OSError as e:
        print(f"OS error: {e}")
    except AttributeError as e:
        print(f"Attribute error: {e}")
'''

def test_train_val_split() -> None:
    """
    Splits a dataset into training, testing, and validation sets based on a predefined ratio.

    This function determines the number of items in each subset of the dataset (training, testing,
    validation) based on the total number of items specified by the global variable `args.n`. It follows
    an 80/10/10 split ratio. If the total number of items is 10 or fewer, the function defaults to using
    1000 items for the training set and evenly divides the remaining 200 items between the test and
    validation sets.

    The function prints the size of each subset to the console.

    Returns:
        tuple: A tuple containing the number of items in the test set, training set, and validation set
        in that order.

    Raises:
        AttributeError: If `args.n` is not defined globally before calling this function.

    Note:
        The function assumes an 80/10/10 split for datasets larger than 10 items. Ensure that `args.n`
        is defined and is an integer representing the total number of images or items in the dataset
        before invoking this function.
    """
    if args.n <= 10:
        print(Fore.RED+
            "At least 10 images are needed for an 80/10/10 dataset. Using the default value of 1000 training images."
        )
        training_set = 800
        test_set = 100
        validation_set = 100
    else:
        total_dataset = args.n
        test_set = int((0.10 * total_dataset) // 1)
        validation_set = int((0.10 * total_dataset) // 1)
        training_set = int((0.80 * total_dataset) // 1)
        print(Fore.BLUE+
            f"\n {total_dataset} images and labels will be split into a 80/10/10 training/test/validation set containing: \n {training_set} training images \n {test_set} test images \n {validation_set} validation images."
        )
    #print(test_set)
    return test_set, training_set, validation_set

def get_classes(data: Dict[int, Dict[str, int]]) -> List[str]:
    """
    Extracts the values of the 'folder' key from each sub-dictionary in the input dictionary.

    Args:
        data (Dict[int, Dict[str, int]]): A dictionary where each key maps to a dictionary
                                          containing 'folder', 'longest_min', and 'longest_max' keys.

    Returns:
        List[str]: A list of values from the 'folder' key in each sub-dictionary.
    """
    return [sub_dict["folder"] for sub_dict in data.values()]


def generate_directory_structure(root_dir: str) -> Dict[str, Any]:
    """
    Generates a nested dictionary representing the directory structure.

    Args:
        root_dir (str): The root directory from which to generate the structure.

    Returns:
        Dict[str, Any]: A nested dictionary representing the directory structure.
    """
    directory_structure = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Split the path into components
        parts = dirpath.split(os.sep)
        # Navigate the nested dictionary to the current directory
        current_level = directory_structure
        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

    return directory_structure


def get_path_from_structure(structure: Dict[str, Any], *path: str) -> Optional[str]:
    """
    Retrieves the relative path from the directory structure.

    Args:
        structure (Dict[str, Any]): The nested dictionary representing the directory structure.
        path (str): The sequence of keys representing the desired path.

    Returns:
        Optional[str]: The relative path as a string if found, otherwise None.
    """
    current_level = structure
    for part in path:
        if part in current_level:
            current_level = current_level[part]
        else:
            return None
    return os.path.join(*path) if isinstance(current_level, dict) else None


def convert_yolo_to_voc(yolo_label_path: str, voc_label_path: str, image_width: int, image_height: int):
    """
    Converts a YOLO label to VOC format.
    
    Args:
        yolo_label_path (str): Path to the YOLO label file.
        voc_label_path (str): Path to save the VOC label file.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    annotation = ET.Element("annotation")
    
    with open(yolo_label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = str(int(class_id))  # Replace with class name if available
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bbox = ET.SubElement(obj, "bndbox")
            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)
            
            ET.SubElement(bbox, "xmin").text = str(x_min)
            ET.SubElement(bbox, "ymin").text = str(y_min)
            ET.SubElement(bbox, "xmax").text = str(x_max)
            ET.SubElement(bbox, "ymax").text = str(y_max)
    
    tree = ET.ElementTree(annotation)
    tree.write(voc_label_path)

def convert_yolo_to_coco(yolo_label_path: str, coco_annotations: List[Dict[str, Any]], image_id: int, category_id: int, image_width: int, image_height: int):
    """
    Converts a YOLO label to COCO format and appends it to the provided COCO annotations list.
    
    Args:
        yolo_label_path (str): Path to the YOLO label file.
        coco_annotations (List[Dict[str, Any]]): List to append the COCO annotations to.
        image_id (int): ID of the image.
        category_id (int): ID of the category.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    with open(yolo_label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            
            annotation = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            }
            coco_annotations.append(annotation)

def convert_labels(root_dir: str, conversion_type: str):
    """
    Converts labels from YOLO format to VOC or COCO format and deletes the old YOLO labels.
    
    Args:
        root_dir (str): The root directory containing 'train', 'test', and 'val' subdirectories.
        conversion_type (str): The target format for conversion ('voc' or 'coco').
    """
    subsets = ['train', 'test', 'val']
    image_ext = '.jpg'  # Assuming images are in .jpg format, change if necessary
    
    for subset in subsets:
        labels_dir = os.path.join(root_dir, subset, 'labels')
        images_dir = os.path.join(root_dir, subset, 'images')
        new_labels_dir = os.path.join(root_dir, subset, 'labels')
        
        # Create the new labels directory if it doesn't exist
        os.makedirs(new_labels_dir, exist_ok=True)
        
        if conversion_type == 'coco':
            coco_annotations = []
            coco_images = []
            coco_categories = [{"id": 1, "name": "class_name"}]  # Replace with actual category names and IDs
            
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):  # Assuming YOLO labels are in .txt format
                yolo_label_path = os.path.join(labels_dir, label_file)
                image_name = label_file.replace('.txt', image_ext)
                image_path = os.path.join(images_dir, image_name)
                
                # Get image dimensions
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        from PIL import Image
                        with Image.open(img_file) as img:
                            image_width, image_height = img.size
                else:
                    continue
                
                new_label_path = os.path.join(new_labels_dir, label_file.replace('.txt', '.xml' if conversion_type == 'voc' else '.json'))
                
                if conversion_type == 'voc':
                    convert_yolo_to_voc(yolo_label_path, new_label_path, image_width, image_height)
                elif conversion_type == 'coco':
                    image_id = len(coco_images) + 1
                    category_id = 1  # Replace with actual category ID lookup if necessary
                    coco_images.append({
                        "id": image_id,
                        "file_name": image_name,
                        "width": image_width,
                        "height": image_height
                    })
                    convert_yolo_to_coco(yolo_label_path, coco_annotations, image_id, category_id, image_width, image_height)
                
                # Delete the old YOLO label
                os.remove(yolo_label_path)
        
        if conversion_type == 'coco':
            coco_output = {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": coco_categories
            }
            coco_output_path = os.path.join(new_labels_dir, 'annotations.json')
            with open(coco_output_path, 'w') as json_file:
                json.dump(coco_output, json_file, indent=4)
        
        # Optionally, remove the old labels directory if it's empty
        if not os.listdir(labels_dir):
            os.rmdir(labels_dir)

def generate() -> None:
    """
    Orchestrates the generation of a dataset by executing a sequence of operations.

    This function performs the following tasks in sequence:
    1. Calls `obj_list` to process object listings, assuming it prepares or manipulates
       some required data or structures.
    2. Calls `mkdir` to presumably create directories needed for dataset storage.
    3. Splits the dataset into training, testing, and validation sets using the `test_train_val_split` function,
       capturing the sizes of each split.
    4. Generates the actual dataset files for the test, validation, and training sets by calling
       `generate_dataset` with the respective sizes and designated folders.

    The function relies on external definitions and side effects from `obj_list`, `mkdir`, and `generate_dataset`.
    These functions are expected to be implemented properly and accessible globally for `generate` to function
    correctly.

    Raises:
        Exception: If any called function (`obj_list`, `mkdir`, `test_train_val_split`, `generate_dataset`)
                   fails, the exception will be propagated upward, detailing the cause of the failure.

    Note:
        The function assumes all directory paths and other necessary environmental settings are correctly
        configured before execution. Errors related to environment misconfigurations or missing resources
        will not be handled within this function.
    """
    obj_list()
    mkdir()
    ttv = test_train_val_split()
    generate_dataset(ttv[0], folder="dataset", split="test")
    generate_dataset(ttv[2], folder="dataset", split="val")
    generate_dataset(ttv[1], folder="dataset", split="train")


def get_classes(data: Dict[int, Dict[str, int]]) -> List[str]:
    """
    Extracts the values of the 'folder' key from each sub-dictionary in the input dictionary.

    Args:
        data (Dict[int, Dict[str, int]]): A dictionary where each key maps to a dictionary
                                          containing 'folder', 'longest_min', and 'longest_max' keys.

    Returns:
        List[str]: A list of values from the 'folder' key in each sub-dictionary.
    """
    return [sub_dict["folder"] for sub_dict in data.values()]


def generate_directory_structure(root_dir: str) -> Dict[str, Any]:
    """
    Generates a nested dictionary representing the directory structure.

    Args:
        root_dir (str): The root directory from which to generate the structure.

    Returns:
        Dict[str, Any]: A nested dictionary representing the directory structure.
    """
    directory_structure = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Split the path into components
        parts = dirpath.split(os.sep)
        # Navigate the nested dictionary to the current directory
        current_level = directory_structure
        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

    return directory_structure


def get_path_from_structure(structure: Dict[str, Any], *path: str) -> Optional[str]:
    """
    Retrieves the relative path from the directory structure.

    Args:
        structure (Dict[str, Any]): The nested dictionary representing the directory structure.
        path (str): The sequence of keys representing the desired path.

    Returns:
        Optional[str]: The relative path as a string if found, otherwise None.
    """
    current_level = structure
    for part in path:
        if part in current_level:
            current_level = current_level[part]
        else:
            return None
    return os.path.join(*path) if isinstance(current_level, dict) else None


def convert_yolo_to_voc(yolo_label_path: str, voc_label_path: str, image_width: int, image_height: int):
    """
    Converts a YOLO label to VOC format.
    
    Args:
        yolo_label_path (str): Path to the YOLO label file.
        voc_label_path (str): Path to save the VOC label file.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    annotation = ET.Element("annotation")
    
    with open(yolo_label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = str(int(class_id))  # Replace with class name if available
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bbox = ET.SubElement(obj, "bndbox")
            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)
            
            ET.SubElement(bbox, "xmin").text = str(x_min)
            ET.SubElement(bbox, "ymin").text = str(y_min)
            ET.SubElement(bbox, "xmax").text = str(x_max)
            ET.SubElement(bbox, "ymax").text = str(y_max)
    
    tree = ET.ElementTree(annotation)
    tree.write(voc_label_path)

def convert_yolo_to_coco(yolo_label_path: str, coco_annotations: List[Dict[str, Any]], image_id: int, category_id: int, image_width: int, image_height: int):
    """
    Converts a YOLO label to COCO format and appends it to the provided COCO annotations list.
    
    Args:
        yolo_label_path (str): Path to the YOLO label file.
        coco_annotations (List[Dict[str, Any]]): List to append the COCO annotations to.
        image_id (int): ID of the image.
        category_id (int): ID of the category.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    with open(yolo_label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            
            annotation = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            }
            coco_annotations.append(annotation)

def convert_labels(root_dir: str, conversion_type: str):
    """
    Converts labels from YOLO format to VOC or COCO format and deletes the old YOLO labels.
    
    Args:
        root_dir (str): The root directory containing 'train', 'test', and 'val' subdirectories.
        conversion_type (str): The target format for conversion ('voc' or 'coco').
    """
    subsets = ['train', 'test', 'val']
    image_ext = '.jpg'  # Assuming images are in .jpg format, change if necessary
    
    for subset in subsets:
        labels_dir = os.path.join(root_dir, subset, 'labels')
        images_dir = os.path.join(root_dir, subset, 'images')
        new_labels_dir = os.path.join(root_dir, subset, 'labels')
        
        # Create the new labels directory if it doesn't exist
        os.makedirs(new_labels_dir, exist_ok=True)
        
        if conversion_type == 'coco':
            coco_annotations = []
            coco_images = []
            coco_categories = [{"id": 1, "name": "class_name"}]  # Replace with actual category names and IDs
            
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):  # Assuming YOLO labels are in .txt format
                yolo_label_path = os.path.join(labels_dir, label_file)
                image_name = label_file.replace('.txt', image_ext)
                image_path = os.path.join(images_dir, image_name)
                
                # Get image dimensions
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        from PIL import Image
                        with Image.open(img_file) as img:
                            image_width, image_height = img.size
                else:
                    continue
                
                new_label_path = os.path.join(new_labels_dir, label_file.replace('.txt', '.xml' if conversion_type == 'voc' else '.json'))
                
                if conversion_type == 'voc':
                    convert_yolo_to_voc(yolo_label_path, new_label_path, image_width, image_height)
                elif conversion_type == 'coco':
                    image_id = len(coco_images) + 1
                    category_id = 1  # Replace with actual category ID lookup if necessary
                    coco_images.append({
                        "id": image_id,
                        "file_name": image_name,
                        "width": image_width,
                        "height": image_height
                    })
                    convert_yolo_to_coco(yolo_label_path, coco_annotations, image_id, category_id, image_width, image_height)
                
                # Delete the old YOLO label
                os.remove(yolo_label_path)
        
        if conversion_type == 'coco':
            coco_output = {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": coco_categories
            }
            coco_output_path = os.path.join(new_labels_dir, 'annotations.json')
            with open(coco_output_path, 'w') as json_file:
                json.dump(coco_output, json_file, indent=4)
        
        # Optionally, remove the old labels directory if it's empty
        if not os.listdir(labels_dir):
            os.rmdir(labels_dir)

def get_classes(data: Dict[int, Dict[str, int]]) -> List[str]:
    """
    Extracts the values of the 'folder' key from each sub-dictionary in the input dictionary.

    Args:
        data (Dict[int, Dict[str, int]]): A dictionary where each key maps to a dictionary
                                          containing 'folder', 'longest_min', and 'longest_max' keys.

    Returns:
        List[str]: A list of values from the 'folder' key in each sub-dictionary.
    """
    return [sub_dict["folder"] for sub_dict in data.values()]

def visualize_yolo_bboxes(image_path: str, yolo_txt_path: str, class_names: list):
    """
    Visualizes YOLO bounding boxes on an image.

    Parameters:
        image_path (str): The path to the image file.
        yolo_txt_path (str): The path to the YOLO text file containing bounding box annotations.
        class_names (list): A list of class names corresponding to the class IDs in the YOLO file.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Load the YOLO annotations
    with open(yolo_txt_path, 'r') as file:
        lines = file.readlines()

    # Parse the YOLO annotations and draw bounding boxes
    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.strip().split())
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
        x_min, y_min = int(x_center - w / 2), int(y_center - h / 2)
        x_max, y_max = int(x_center + w / 2), int(y_center + h / 2)

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, class_names[int(class_id)], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image with bounding boxes
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def visualize_random_image_from_set(set_name: str, class_names: list):
    """
    Visualizes a random image and its corresponding YOLO annotations from a specified set.

    Parameters:
        set_name (str): The name of the set ('train', 'test', or 'validation').
        class_names (list): A list of class names corresponding to the class IDs in the YOLO file.
    """
    set_path = os.path.join('/home/nalkuq/cmm/dataset', set_name)
    images_path = os.path.join(set_path, 'images')
    labels_path = os.path.join(set_path, 'labels')

    # Choose a random image file
    image_file = random.choice(os.listdir(images_path))
    image_path = os.path.join(images_path, image_file)

    # Construct the corresponding label file path
    label_file = os.path.splitext(image_file)[0] + '.txt'
    yolo_txt_path = os.path.join(labels_path, label_file)

    # Visualize the image with bounding boxes
    visualize_yolo_bboxes(image_path, yolo_txt_path, class_names)

def main():
    class_names = obj_list()
    class_names = get_classes(data)

    print("Visualizing a random image from the training set:")
    visualize_random_image_from_set('train', class_names)

    print("Visualizing a random image from the test set:")
    visualize_random_image_from_set('test', class_names)

    print("Visualizing a random image from the validation set:")
    visualize_random_image_from_set('val', class_names)

if __name__ == "__main__":
    generate()
# Generate the directory structure for the dataset directory
root_directory = "dataset"
directory_structure = generate_directory_structure(root_directory)

# Retrieve the paths
train_path = get_path_from_structure(directory_structure, "dataset", "train")
train_images_path = get_path_from_structure(
    directory_structure, "dataset", "train", "images"
)
train_labels_path = get_path_from_structure(
    directory_structure, "dataset", "train", "labels"
)
test_path = get_path_from_structure(directory_structure, "dataset", "test")
test_images_path = get_path_from_structure(
    directory_structure, "dataset", "test", "images"
)
test_labels_path = get_path_from_structure(
    directory_structure, "dataset", "test", "labels"
)
val_path = get_path_from_structure(directory_structure, "dataset", "val")
val_images_path = get_path_from_structure(
    directory_structure, "dataset", "val", "images"
)
val_labels_path = get_path_from_structure(
    directory_structure, "dataset", "val", "labels"
)

# Create a list of classes and the length of classes
data = obj_list()
class_names = get_classes(data)
class_number = len(class_names)

if args.format == "yolo":
    print(Fore.BLUE+"\n Saving labels in YOLO format \n")
    # Define the content of the classes.yaml file
    classes_yaml_content = OrderedDict(
        [
            ("path", os.path.abspath(root_directory)),
            ("train", os.path.abspath(train_path)),
            ("test", os.path.abspath(test_path)),
            ("val", os.path.abspath(val_path)),
            ("nc", class_number),
            ("names", class_names),
        ]
    )

    # Custom YAML representer for OrderedDict to ensure correct order
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    def ordered_dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    yaml.add_representer(OrderedDict, ordered_dict_representer, Dumper=NoAliasDumper)
    # Path to save the YAML file
    yaml_file_path = os.path.join(root_directory, "classes.yaml")
    # Save the content to classes.yaml
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(
            classes_yaml_content,
            yaml_file,
            Dumper=NoAliasDumper,
            default_flow_style=False,
        )
    # Print results
    print(Fore.BLUE+f"YAML file with name classes.yaml saved to {yaml_file_path}")
    print(class_names, class_number)
elif args.format == "voc":
    print(Fore.BLUE+"\n Saving labels in VOC format \n")
    root_directory = 'dataset'
    conversion_type = 'voc'  
    convert_labels(root_directory, conversion_type)
    try: 
        os.remove("classes.yaml")
    except: 
        pass
    labelmap_path = os.path.join(root_directory, "labelmap.txt")
    with open(labelmap_path, "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"id: {i + 1}\n")
            f.write(f"name: \"{name}\"\n")
    print(Fore.GREEN+"\n Labels have been saved. A labelmap.txt file has been saved in /dataset")
elif args.format == "coco":
    print(Fore.GREEN+"\n Saving labels in COCO format\n")
    root_directory = 'dataset'
    conversion_type = 'coco'
    convert_labels(root_directory, conversion_type)
    try: 
        os.remove("classes.yaml")
        os.remove("labelmap.txt")
    except: 
        pass
    print(Fore.BLUE+"\n Labels have been saved. An annotation file has been saved in /dataset")
else:
    pass

main()


