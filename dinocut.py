import os
import torch
import random
import supervision as sv
import argparse
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import math
import cv2 as cv
from numpy import ndarray
import yaml
from tqdm import tqdm
import time
from colorama import init, Fore, Back, Style
import warnings
import emoji
import shutil
from pyfiglet import Figlet

f = Figlet(font="slant")
print(f.renderText("DINOcut"))

# Hides pytorch warnings regarding Gradient and other cross-depencies, which are pinned in DINOcut
warnings.filterwarnings("ignore")
init()


def check_path(path):
    if os.path.isfile(path):
        return path
    elif os.path.isdir(path):
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
    else:
        raise ValueError(
            "The provided path is neither a file nor a directory. Ensure that your source image path in DINOcut.yaml is correct"
        )


def print_emoji_line(emoji_code: str, repeat_count: int) -> None:
    """
    Print an emoji multiple times on one line based on the provided emoji code.

    Parameters:
    emoji_code (str): The code for the emoji to be printed, e.g., ":T-Rex:".
    repeat_count (int): The number of times the emoji should be printed.

    Returns:
    None
    """
    emoji_text = emoji.emojize(emoji_code)
    print("\n", emoji_text * repeat_count, "\n")


print_emoji_line(":T-Rex:", 1)


parser = argparse.ArgumentParser(
    description="GroundingDINO and SAM for mask extraction."
)
# add optional arguments
parser.add_argument(
    "-src",
    "-src_dir",
    type=os.path.abspath,
    help="the source directory containing your images that you will use to generate masks",
    default=os.path.abspath("starter_dataset"),
)

parser.add_argument(
    "-d_weights",
    "-dino_weights",
    type=os.path.abspath,
    help="the directory containing your GroundingDINO weights",
    default=os.path.abspath("GroundingDINO/weights"),
)


parser.add_argument(
    "-s_weights",
    "-sam_weights",
    type=os.path.abspath,
    help="the directory containing your SAM weights",
    default=os.path.abspath("sam/weights"),
)

args = parser.parse_args()

if args.src:
    PATH_MAIN = args.src
    print(
        Fore.RED + f"\n No source directory given. Main Path set to:",
        Fore.BLUE + f"{PATH_MAIN}",
        Fore.RED + "Please use python3 dino_sam.py -h to learn more.\n",
    )
else:
    PATH_MAIN = os.path.abspath("starter_dataset")
    print(
        Back.GREEN
        + f"\n No source directory given. Main Path set to {PATH_MAIN}. Please use python3 dino_sam.lspy -h to learn more\n"
    )


def cuda_enabled() -> str:
    """
    Checks if CUDA is available on the system and prints relevant information.

    Returns:
        str: 'cuda' if CUDA is available, otherwise 'cpu'.

    Notes:
        The function prints whether CUDA is available, the count of CUDA devices,
        and the current active CUDA device index. If CUDA is not available,
        it advises checking the CUDA installation.
    """
    try:
        with tqdm(total=3, desc=Fore.BLUE + "Checking CUDA availability") as pbar:
            if torch.cuda.is_available():
                pbar.set_postfix(step=Fore.BLUE + "Checking if CUDA is available")
                time.sleep(1)  # Simulate time delay
                pbar.update(1)

                pbar.set_postfix(step=Fore.BLUE + "Counting CUDA devices")
                device_count = torch.cuda.device_count()
                time.sleep(1)  # Simulate time delay
                pbar.update(1)

                pbar.set_postfix(step=Fore.BLUE + "Getting current CUDA device index")
                current_device = torch.cuda.current_device()
                time.sleep(1)  # Simulate time delay
                pbar.update(1)

                print("\n")
                print(Fore.GREEN + "\tCUDA is available.")
                print(Fore.GREEN + f"\tNumber of CUDA devices: {device_count}")
                print(Fore.GREEN + f"\tCurrent CUDA device index: {current_device}")
                return "cuda"
            else:
                pbar.set_postfix(step="CUDA not available, using CPU")
                time.sleep(1)  # Simulate time delay
                pbar.update(1)

                print(Fore.RED + "\tCUDA is not available. Using CPU instead.")
                return "cpu"
    except Exception as e:
        print(Fore.RED + "\tAn error occurred while checking CUDA: ", str(e))
        print(
            Fore.RED
            + "\tCUDA is not detected. Ensure your CUDA is up-to-date with nvidia-smi"
        )
        return "cpu"


# Init CUDA
print_emoji_line(":T-Rex:", 2)
device = cuda_enabled()


def load_configuration(yaml_path: str) -> Dict[str, Any]:
    """
    Load and parse configuration from a YAML file.

    Parameters:
    - yaml_path (str): The path to the YAML configuration file.

    Returns:
    - Dict[str, Any]: A dictionary with all configurations loaded from the YAML.

    Raises:
    - IOError: If there's an error opening or reading the YAML file.
    - ValueError: If the YAML file is empty or improperly formatted.

    Example:
    Assume a YAML file 'config.yaml' with the following content:
    ---
    paths:
      grounding_dino_checkpoint_path: "${d_weights}/groundingdino_swint_ogc.pth"
      grounding_dino_config_path: "config/GroundingDINO_SwinT_OGC.py"
    model_configs:
      sam:
        encoder_version: "vit_h"

    Using the function:
    >>> yaml_path = 'config.yaml'
    >>> config = load_configuration(yaml_path)
    >>> print(config['paths']['grounding_dino_checkpoint_path'])
    /usr/local/d_weights/groundingdino_swint_ogc.pth
    """
    try:
        with tqdm(
            total=3, desc=Fore.BLUE + "Loading DINOcut.yaml Configuration file"
        ) as pbar:
            # Load YAML content
            pbar.set_postfix(step="Opening YAML file")
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
            pbar.update(1)

            # Check if the YAML file was empty or improperly formatted
            pbar.set_postfix(step="Parsing YAML file")
            if config is None:
                raise ValueError(
                    "The YAML file is empty or the contents are not in valid YAML format."
                )
            pbar.update(1)

            # Resolve environment variables in paths if they are included
            pbar.set_postfix(step="Resolving environment variables")
            if "paths" in config:
                for key, value in config["paths"].items():
                    # Replace environment variables in paths with actual values
                    config["paths"][key] = os.path.expandvars(value)
            pbar.update(1)

    except Exception as e:
        raise IOError(
            Fore.RED
            + f"An error occurred while reading or parsing the YAML file: {str(e)}"
        )

    return config


print_emoji_line(":T-Rex:", 3)
# Example usage
yaml_path = "dinocut_config.yaml"
config = load_configuration(yaml_path)


# Environment variables
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
    args.d_weights, config["paths"]["grounding_dino_checkpoint_path"]
)
GROUNDING_DINO_CONFIG_PATH = config["paths"]["grounding_dino_config_path"]

SAM_CHECKPOINT_PATH = os.path.join(
    args.s_weights, config["paths"]["sam_checkpoint_path"]
)
SAM_ENCODER_VERSION = config["model_configs"]["sam"]["encoder_version"]


grounding_dino_model = Model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
)
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
    device=device
)
sam_predictor = SamPredictor(sam)

# Need to auomate this through and debug dinocut-config.yaml. Make sure to debug and pyut try/except loops. There is a problem when there is no detection.
SOURCE_IMAGE_PATH = config["image_settings"]["source_image_path"]


CLASSES = config["image_settings"]["classes"]
CLASSES = [s.lower() for s in CLASSES]
BOX_TRESHOLD = config["image_settings"]["thresholds"]["box"]
TEXT_TRESHOLD = config["image_settings"]["thresholds"]["text"]


print_emoji_line(":T-Rex:", 4)
print(Fore.BLUE + "\nChecking Config Settings:")
print(
    Fore.GREEN + "\n\tGrounding DINO Checkpoint Path:",
    Fore.RED + config["paths"]["grounding_dino_checkpoint_path"],
)
print(
    Fore.GREEN + "\tSAM Checkpoint Path:",
    Fore.RED + config["paths"]["sam_checkpoint_path"],
)
print(
    Fore.GREEN + "\tBox Threshold:",
    Fore.RED + str(config["image_settings"]["thresholds"]["box"]),
)
print(
    Fore.GREEN + "\tBox Threshold:",
    Fore.RED + str(config["image_settings"]["thresholds"]["text"]),
)
print(Fore.GREEN + f"\tYour text prompt is:", Fore.RED + f"{CLASSES}")


print(Fore.GREEN + "\tImage Source Path:", Fore.RED + f"{SOURCE_IMAGE_PATH}")
print_emoji_line(":T-Rex:", 5)


def enhance_class_name(class_names: List[str]) -> List[str]:
    """
    Enhances a list of class names by formatting them to indicate a collection of each class.

    Parameters:
        class_names (List[str]): A list of strings representing class names.

    Returns:
        List[str]: A list containing the enhanced names, where each name is prefixed with "all"
                   and suffixed with an "s" to indicate pluralization.

    Example:
        >>> enhance_class_name(["cat", "dog"])
        ["all cats", "all dogs"]

    Description:
        This function takes each class name in the provided list and formats it to reflect
        a collection of items of that class by adding 'all ' before the name and appending
        an 's' at the end. This is useful for creating labels or descriptions that need to
        refer to multiple instances of each class.
    Citation:
         Function modified from 2024 Grounded SAM Release by Ren et. al,
            @misc{ren2024grounded,
                    title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks},
                    author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
                    year={2024},
                    eprint={2401.14159},
                    archivePrefix={arXiv},
                    primaryClass={cs.CV}
                    }
    """
    return [f"all {class_name}s" for class_name in class_names]


def dino_detection(
    image: np.ndarray,
    class_names: List[str],
    box_threshold: float,
    text_threshold: float,
) -> List[Tuple[Any, ...]]:
    """
    Performs object detection on an image using specific class names, and detection thresholds.

    Parameters:
        image (np.ndarray): The image on which object detection will be performed.
        class_names (List[str]): The names of the classes to detect.
        box_threshold (float): The threshold for box detection.
        text_threshold (float): The threshold for text detection.

    Returns:
        List[Tuple[Any, ...]]: A list of detection results, where each element is a tuple containing
                               the detection details.

    Example Usage:
        # Assuming you have an image loaded as a numpy array and the following settings:
        # - `class_names`: A list of strings representing class names, e.g., ['cat', 'dog'].
        # - `box_threshold`: A float value for box detection threshold, e.g., 0.5.
        # - `text_threshold`: A float value for text detection threshold, e.g., 0.7.

        image = np.random.rand(224, 224, 3)  # Random image for demonstration
        class_names = ['cat', 'dog']
        box_threshold = 0.5
        text_threshold = 0.7

        detections = dino_detection(image, class_names, box_threshold, text_threshold)
        print(detections)
    Citation:
         Function modified from 2024 Grounded SAM Release by Ren et. al,
            @misc{ren2024grounded,
                    title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks},
                    author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
                    year={2024},
                    eprint={2401.14159},
                    archivePrefix={arXiv},
                    primaryClass={cs.CV}
                    }
    """
    # Assuming grounding_dino_model is predefined and has a predict_with_classes method
    enhanced_classes = enhance_class_name(class_names)
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhanced_classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    return detections


def dino_display_image(
    image: np.ndarray,
    detections: List[Tuple[Any, Any, float, int, Any]],
    CLASSES: List[str],
) -> None:
    """
    Annotates an image with detection boxes and labels, then displays it.

    Parameters:
        image (np.ndarray): The image to annotate. This should be a NumPy array.
        detections (List[Tuple[Any, Any, float, int, Any]]): A list of detections, each a tuple containing:
                                                            (_, _, confidence, class_id, _). It is assumed that
                                                            each tuple provides _, _, confidence score, class ID, and _
                                                            in that order.
        CLASSES (List[str]): A list of class names corresponding to class IDs.

    Example Usage:
        # Assuming you have the following data:
        # - `image`: A numpy array representing the image data.
        # - `detections`: A list of tuples, each representing a detection.
        #   Example of a detection tuple: (0, 0, 0.9, 1, 0)
        # - `CLASSES`: A list of string labels for classes, e.g., ['cat', 'dog']

        image = np.random.rand(800, 600, 3)  # Random image for demonstration
        detections = [(0, 0, 0.9, 1, 0), (0, 0, 0.75, 0, 0)]
        CLASSES = ['cat', 'dog']
        dino_display_image(image, detections, CLASSES)
    Citation:
         Function modified from 2024 Grounded SAM Release by Ren et. al,
            @misc{ren2024grounded,
                    title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks},
                    author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
                    year={2024},
                    eprint={2401.14159},
                    archivePrefix={arXiv},
                    primaryClass={cs.CV}
                    }
    """
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:.2f}"
        for _, _, confidence, class_id, _ in detections
    ]
    annotated_frame = box_annotator.annotate(
        scene=image.copy(), detections=detections, labels=labels
    )
    sv.plot_image(annotated_frame, (16, 16))


def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def show_sam_detections(
    image: np.ndarray,
    detections: List[Tuple[Any, Any, float, int, Any]],
    CLASSES: List[str],
) -> None:
    """
    Annotates an image with detection masks and boxes based on given detections, then displays the image.
    Also, displays a grid of detection categories based on the number of detections.

    Parameters:
        image (np.ndarray): The image on which annotations are to be made.
        detections (List[Tuple[Any, Any, float, int, Any]]): A list of detections, each a tuple containing
                                                            (_, _, confidence, class_id, _).
        CLASSES (List[str]): A list of class names corresponding to class IDs.

    Example Usage:
        # Assuming you have an image loaded as a numpy array and the following detections:
        # Detections format: (x, y, confidence, class_id, mask)
        # - `CLASSES`: ['cat', 'dog', 'bird']

        image = np.random.rand(800, 600, 3)  # Random image for demonstration
        detections = [
            (50, 50, 0.9, 0, np.zeros((800, 600))),
            (150, 150, 0.85, 1, np.zeros((800, 600)))
        ]
        CLASSES = ['cat', 'dog', 'bird']
        show_sam_detections(image, detections, CLASSES)
    Citation:
         Function modified from 2024 Grounded SAM Release by Ren et. al,
            @misc{ren2024grounded,
                    title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks},
                    author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
                    year={2024},
                    eprint={2401.14159},
                    archivePrefix={arXiv},
                    primaryClass={cs.CV}
                    }
    """
    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    # Generate labels for each detection
    labels = [
        f"{CLASSES[class_id]} {confidence:.2f}"
        for _, _, confidence, class_id, _ in detections
    ]

    # Annotate image with masks and boxes
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    # Display the annotated image
    sv.plot_image(annotated_image, (16, 16))

    # Calculate grid size based on the number of detections
    grid_size_dimension = math.ceil(math.sqrt(len(detections)))

    # Extract class titles for each detection
    titles = [
        CLASSES[d[3]] for d in detections
    ]  # Adjust this if the structure of detections differs


def save_inverted_masks(detections: List[np.ndarray]) -> None:
    """
    Saves inverted color masks from a list of numpy arrays to separate PNG files.

    Each mask in the input list is processed by converting its boolean values to uint8,
    inverting its colors (object becomes black, background becomes white), and saving it
    as a PNG file with a filename indicating its order in the list.

    Parameters:
        detections (List[np.ndarray]): A list of 2D numpy arrays representing boolean masks.

    Returns:
        None: This function does not return a value but saves files to the disk.
    """
    for id, mask in enumerate(detections):
        # Convert the boolean mask to uint8 by multiplying by 255
        id = time.time()
        mask_to_save = (mask * 255).astype(np.uint8)
        # Invert the mask colors: foreground (object) becomes black, background becomes white
        inverted_mask = 255 - mask_to_save
        mask_filename = f"mask_{id}.png"
        # Save the inverted mask using cv2.imwrite
        cv2.imwrite(mask_filename, inverted_mask)
        print(f"Saved inverted mask to {mask_filename}")


def apply_mask(main_img: ndarray, mask_img: ndarray) -> ndarray:
    """
    Applies a mask to an image, extracting the features corresponding to the non-masked areas
    and placing them on a white background.

    Parameters:
        main_img (ndarray): The main image to which the mask will be applied. This should be a color image (3 channels).
        mask_img (ndarray): The mask image, where non-zero (typically 255) areas represent the mask. Should be single channel.

    Returns:
        ndarray: The resulting image after applying the mask. Features from the main image where the mask is non-zero
                 are placed onto a white background, and the rest are kept as is from the main image.

    Example Usage:
        # Assume `main_img` is an image loaded into a numpy array and `mask_img` is a mask image (also a numpy array).
        # Both images should be of the same dimensions.

        main_img = cv2.imread('path_to_image.jpg')
        mask_img = cv2.imread('path_to_mask.jpg', cv2.IMREAD_GRAYSCALE)  # Mask image in grayscale

        result_img = apply_mask(main_img, mask_img)
        cv2.imshow('Masked Image', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    Note:
        - The mask image `mask_img` should be a grayscale image where the areas of interest are marked in white (255),
          and the rest should be black (0). The function will invert this mask internally.
    """
    # Invert the mask to use black (0) areas for extraction (black becomes white and vice versa)
    inverted_mask = cv2.bitwise_not(mask_img)

    # Extract features where the mask is black (now white in the inverted mask)
    feature = cv2.bitwise_and(main_img, main_img, mask=inverted_mask)

    # Create a white background image of the same size
    background = np.ones_like(main_img) * 255

    # Where the original mask is 255 (background), use the white background, else use extracted feature
    # Since we inverted the mask, use the original mask to determine where to apply the background
    masked_feature = np.where(mask_img[..., None] == 0, feature, background).astype(
        np.uint8
    )

    return masked_feature


# this is a problem when you attempt to upload a directory.
# The problem originates with the SOURCE_IMAGE Path Variable
# Need to change this function. Work on this asap.
def create_training_image(path):
    main_image_path = path
    mask_directory = config["paths"]["home_directory"]

    # Load the main image
    main_img = cv2.imread(main_image_path)

    # Ensure the main image loaded correctly
    if main_img is None:
        raise FileNotFoundError(
            "The main image could not be loaded. Check the file path."
        )
    # Process each mask
    for mask_filename in os.listdir(mask_directory):
        if mask_filename.endswith(".png"):
            mask_img = cv2.imread(
                os.path.join(mask_directory, mask_filename), cv2.IMREAD_GRAYSCALE
            )
            # Ensure the mask image loaded correctly
            if mask_img is None:
                raise FileNotFoundError(
                    f"The mask image {mask_filename} could not be loaded. Check the file path."
                )

            # Apply mask and create new image
            feature_img = apply_mask(main_img, mask_img)

            # Save the new image
            feature_img_path = f"{mask_filename[:-4]}.jpg"
            cv2.imwrite(feature_img_path, feature_img)
            print(f"Saved {feature_img_path}")


def dinocut_generate():
    # load image
    print(Fore.BLUE + "\nDINOCUT is detecting instances of:", Fore.GREEN + f"{CLASSES}")
    path = check_path(SOURCE_IMAGE_PATH)
    if isinstance(path, str):
        image = cv2.imread(path)
        detections = dino_detection(image, CLASSES, BOX_TRESHOLD, TEXT_TRESHOLD)
        print_emoji_line(":T-Rex:", 6)
        print(
            Fore.BLUE + "\nDINOCUT is segmenting all detections for:",
            Fore.GREEN + f"{CLASSES}",
        )
        print_emoji_line(":T-Rex:", 7)
        try:
            dino_display_image(image, detections, CLASSES)
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy,
            )
            show_sam_detections(image, detections, CLASSES)
            save_inverted_masks(detections.mask)
            create_training_image(path)
            # need to fix this so that it checks/passes if the directory is already named. Also need to make a images/masks folder.
            directory = f"{path[:-4]}"
            try:
                os.mkdir(directory)
                print(f"Making a directory named {directory} to save masks and labels.")
            except:
                pass
            # change to config file home directory. May have to also change argparse.
            os.chdir("/home/nalkuq/cmm")
            cwd = os.getcwd()
            for files in os.listdir(cwd):
                if files.endswith(".png"):
                    shutil.move(files, directory)
                if files.endswith(".jpg"):
                    shutil.move(files, directory)
                else:
                    pass
        except:
            print(
                Fore.RED + f"\nDinoCut was unable to find any instances of:",
                Fore.BLUE + f"{CLASSES}",
                Fore.RED
                + "\nPlease alter the prompt, box threshold, or text threshold in dincut_config.yaml.",
            )
    if isinstance(path, list):
        for file in path:
            path = file
            image = cv2.imread(path)
            detections = dino_detection(image, CLASSES, BOX_TRESHOLD, TEXT_TRESHOLD)
            print_emoji_line(":T-Rex:", 6)
            print(
                Fore.BLUE + "\nDINOCUT is segmenting all detections for:",
                Fore.GREEN + f"{CLASSES}",
            )
            print_emoji_line(":T-Rex:", 7)
            try:
                dino_display_image(image, detections, CLASSES)
                detections.mask = segment(
                    sam_predictor=sam_predictor,
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    xyxy=detections.xyxy,
                )
                show_sam_detections(image, detections, CLASSES)
                mask = save_inverted_masks(detections.mask)
                create_training_image(path)
                directory = f"{path[:-4]}"
                try:
                    os.mkdir(directory)
                    print(
                        f"Making a directory named {directory} to save masks and labels."
                    )
                except:
                    pass
                # change to config file home directory. May have to also change argparse.
                os.chdir("/home/nalkuq/cmm")
                cwd = os.getcwd()
                for files in os.listdir(cwd):
                    if files.endswith(".png"):
                        shutil.move(files, directory)
                    if files.endswith(".jpg"):
                        shutil.move(files, directory)
                    else:
                        pass
            except:
                print(
                    Fore.RED + f"\nDinoCut was unable to find any instances of:",
                    Fore.BLUE + f"{CLASSES}",
                    Fore.RED
                    + "\nPlease alter the prompt, box threshold, or text threshold in dincut_config.yaml.",
                )


if __name__ == "__main__":
    dinocut_generate()
    os.system(
        "python3 scripts/selector.py --directory /home/nalkuq/cmm/starter_dataset --target-directory /home/nalkuq/cmm/data"
    )
    os.system(
        f'python3 scripts/synthetic.py -n {config["image_settings"]["number"]} -io {config["image_settings"]["image_overlap"]} -max_obj {config["image_settings"]["max_obj"]} -min {config["image_settings"]["min_size"]} -max {config["image_settings"]["max_size"]} -erase {config["image_settings"]["erase"]} -format {config["image_settings"]["format"]}'
    )
