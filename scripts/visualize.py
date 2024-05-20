import cv2
import matplotlib.pyplot as plt

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

# Example usage:
class_names = ['caveg', 'spongebob squarepants', 'ulus', 'semilunar knife', 'tops', 'endblades' ]  # replace with your actual class names
visualize_yolo_bboxes('dataset/val/images/02024_05_20-07_20_04_PM.jpg', 'dataset/val/labels/02024_05_20-07_20_04_PM.txt', class_names)
