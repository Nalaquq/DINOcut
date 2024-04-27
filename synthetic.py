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
from typing import List

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
    default=1000,
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
    default=150,
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
    default="config.yaml",
)
args = parser.parse_args()

if args.src:
    PATH_MAIN = args.src
else:
    PATH_MAIN = os.path.abspath("data")
    print(
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
    folder_count = len(next(os.walk(PATH_MAIN))[1])
    for f in range(folder_count):
        obj_dict[f] = {
            "folder": next(os.walk(PATH_MAIN))[1][f],
            "longest_min": args.min,
            "longest_max": args.max,
        }
    # delete the "background images" since it does not have the same directory structure
    del obj_dict[1]
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
        Original Conceptual Credit: @InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
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
        Original Conceptual Credit: @InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
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
    print("Background and Object Transformations loaded successfully.")
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
        Original Conceptual Credit: @InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
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
        Original Conceptual Credit: @InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
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
        Original Conceptual Credit: @InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
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
    mask_comp: np.ndarray, obj_areas: list, overlap_degree: float = 0.3
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
        Original Conceptual Credit: @InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
    """
    obj_ids = np.unique(mask_comp).astype(np.uint8)[
        1:-1
    ]  # Skip the background (typically 0) and exclude the last ID if unused
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
        Original Conceptual Credit: @InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
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
        obj_idx = np.random.randint(len(obj_dict))  # + 1

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


def create_yolo_annotations(mask_comp: np.ndarray, labels_comp: np.ndarray) -> list:
    """
    Generate YOLO format annotations for objects identified in a mask with associated labels.

    Parameters:
        mask_comp (np.ndarray): A 2D array where each pixel's integer value represents the object ID.
        labels_comp (np.ndarray): An array containing the labels corresponding to each object ID in mask_comp.

    Returns:
        list: A list of lists, where each inner list contains normalized bounding box information
              and the label for an object in YOLO format ([class_id, x_center, y_center, width, height]).

    Notes:
        The function expects `mask_comp` to be a grayscale image where different values correspond to
        different objects. `labels_comp` is expected to be an array where each entry corresponds to a label
        for the objects identified in `mask_comp`. The function calculates the bounding box coordinates for
        each object, normalizes these coordinates relative to the dimensions of `mask_comp`, and formats
        them according to the YOLO annotation standard.
    Citation:
        Function modified from https://github.com/alexppppp/synthetic-dataset-object-detection.
        Original Conceptual Credit: @InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
    """
    comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]

    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
    masks = mask_comp == obj_ids[:, None, None]

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

        annotations_yolo.append(
            [
                labels_comp[i] - 1,
                round(xc / comp_w, 5),
                round(yc / comp_h, 5),
                round(w / comp_w, 5),
                round(h / comp_h, 5),
            ]
        )

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
        Original Conceptual Credit: @InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
    """
    time_start = time.time()
    timing = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    for j in tqdm(range(imgs_number)):
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

        img_comp = cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(folder, split, "images/{}" + timing + ".jpg").format(j),
            img_comp,
        )

        annotations_yolo = create_yolo_annotations(mask_comp, labels_comp)
        for i in range(len(annotations_yolo)):
            with open(
                os.path.join(folder, split, "labels/{}" + timing + ".txt").format(j),
                "a",
            ) as f:
                f.write(" ".join(str(el) for el in annotations_yolo[i]) + "\n")
    time_end = time.time()
    time_total = round(time_end - time_start)
    time_per_img = round((time_end - time_start) / imgs_number, 1)

    print(
        "Generation of {} synthetic images is completed. It took {} seconds, or {} seconds per image".format(
            imgs_number, time_total, time_per_img
        )
    )
    print("Images are stored in '{}'".format(os.path.join(folder, split, "images")))
    print(
        "Annotations are stored in '{}'".format(os.path.join(folder, split, "labels"))
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
        print("\n\n Checking Project Paths:")
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
        print("\n\t Home Directory: '{}' ".format(home))
        print("\n\t Data For Generation: '{}'".format(PATH_MAIN))
    except:
        dataset = os.path.join(home, "dataset")
        print("\n\t Home Directory:'{}' ".format(home))
        print("\n\t Data For Generation: '{}' ".format(PATH_MAIN))
        print("\n\t Generated Datasets Stored in: '{}' ".format(dataset))
        # insert if statement for argparse libary that runs clean remove existing dataset -del if set to true. Default will be false..
        total_number = []
        if args.erase == True:
            print("\n\tThe Erase function has been enabled \n")
            print("\n Removing Old Datasets from: {}".format(dataset))
            for root, dirs, files in os.walk(dataset):
                for name in tqdm(files):
                    total_number.append(files)
                    if name.endswith(".jpg"):
                        selected_files = os.path.join(root, name)
                        os.remove(selected_files)
                    if name.endswith(".txt"):
                        selected_files = os.path.join(root, name)
                        os.remove(selected_files)
            print("\n{} files were deleted.".format(len(total_number)))
        else:
            for root, dirs, files in os.walk(dataset):
                for name in files:
                    if name.endswith(".jpg"):
                        selected_files = os.path.join(root, name)
                    if name.endswith(".txt"):
                        selected_files = os.path.join(root, name)
            print("\n\t {} files were deleted.".format(len(total_number)))
            print(
                "\n\t There are {} labels and images in this dataset.".format(
                    len(selected_files)
                )
            )
            print("\n Beginning Data Generation\n")
            pass


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
        print(
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
        print(
            f"\n {total_dataset} images and labels will be split into a 80/10/10 training/test/validation set containing: \n {training_set} training images \n {test_set} test images \n {validation_set} validation images."
        )
    print(test_set)
    return test_set, training_set, validation_set


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


generate()
