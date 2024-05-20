import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import yaml
import argparse
from typing import List, Dict, Any, Tuple


class ImageMaskViewer:
    def __init__(self, root: tk.Tk, config_file: str, initial_directory: str = None, target_directory: str = None) -> None:
        """
        Initialize the ImageMaskViewer application.

        Args:
            root (tk.Tk): The root window of the Tkinter application.
            config_file (str): Path to the configuration YAML file.
            initial_directory (str, optional): The initial directory to load images and masks from.
            target_directory (str, optional): The directory to move the images and masks to after classification.
        """
        self.root = root
        self.root.title("Image and Mask Viewer")
        self.directory = initial_directory or ""
        self.target_directory = target_directory
        self.image_files: List[str] = []
        self.image_ids: Dict[str, int] = {}
        self.current_index = 0
        self.config_file = config_file
        self.class_labels: List[str] = self.load_class_labels()
        self.selected_class = tk.StringVar()
        self.saved_files: Dict[str, List[Tuple[str, str]]] = {label: [] for label in self.class_labels}

        self.load_button = tk.Button(self.root, text="Load Directory", command=self.load_directory)
        self.load_button.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side="left", padx=10)

        self.mask_label = tk.Label(self.root)
        self.mask_label.pack(side="right", padx=10)

        self.id_label = tk.Label(self.root, text="Image ID:")
        self.id_label.pack()

        self.id_text = tk.Entry(self.root, state='readonly')
        self.id_text.pack()

        self.radio_frame = tk.Frame(self.root)
        self.radio_frame.pack(pady=10)

        for label in self.class_labels:
            rb = tk.Radiobutton(self.radio_frame, text=label, variable=self.selected_class, value=label)
            rb.pack(anchor="w")

        self.delete_button = tk.Button(self.root, text="Delete Pair", command=self.delete_pair)
        self.delete_button.pack()

        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side="left", padx=10)

        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(side="right", padx=10)

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side="bottom", fill="x")

        self.exit_button = tk.Button(self.bottom_frame, text="Save and Exit", command=self.save_and_exit)
        self.exit_button.pack(pady=20)

        if initial_directory:
            self.load_images_from_directory(initial_directory)

    def load_class_labels(self) -> List[str]:
        """
        Load class labels from the configuration YAML file.

        Returns:
            List[str]: A list of class labels.
        """
        with open(self.config_file, "r") as file:
            config: Dict[str, Any] = yaml.safe_load(file)
            return config.get("image_settings", {}).get("classes", [])

    def load_directory(self) -> None:
        """
        Open a file dialog to select a directory and load all .jpg image files and corresponding .png mask files from it and its subdirectories.
        """
        self.directory = filedialog.askdirectory()
        if not self.directory:
            return
        self.load_images_from_directory(self.directory)

    def load_images_from_directory(self, directory: str) -> None:
        """
        Load all .jpg image files and corresponding .png mask files from the specified directory and its subdirectories.

        Args:
            directory (str): The directory to load images and masks from.
        """
        self.image_files = []
        self.image_ids = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(root, file)
                    mask_path = image_path.replace('.jpg', '.png')
                    if os.path.exists(mask_path):
                        self.image_files.append(image_path)
        self.image_ids = {f: idx + 1 for idx, f in enumerate(self.image_files)}  # Assign a sequential ID to each image file
        self.current_index = 0
        self.show_image()

    def show_image(self) -> None:
        """
        Display the current image and its corresponding mask side by side.
        """
        if not self.image_files:
            return

        image_file = self.image_files[self.current_index]
        mask_file = image_file.replace('.jpg', '.png')

        if not os.path.exists(mask_file):
            messagebox.showerror("Error", f"Mask file {mask_file} not found")
            return

        image = Image.open(image_file)
        mask = Image.open(mask_file)

        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        mask = mask.resize((300, 300), Image.Resampling.LANCZOS)

        self.image_tk = ImageTk.PhotoImage(image)
        self.mask_tk = ImageTk.PhotoImage(mask)

        self.image_label.config(image=self.image_tk)
        self.mask_label.config(image=self.mask_tk)

        # Display the unique ID of the current image
        image_id = self.image_ids[image_file]
        self.id_text.config(state='normal')
        self.id_text.delete(0, tk.END)
        self.id_text.insert(0, str(image_id))
        self.id_text.config(state='readonly')

    def save_current_pair(self) -> bool:
        """
        Save the current image and its corresponding mask to the selected class list.
        If no class is selected, show a warning message.

        Returns:
            bool: True if the pair is saved successfully, False otherwise.
        """
        selected_class = self.selected_class.get()
        if not selected_class:
            messagebox.showwarning("Warning", "Please select a class before proceeding.")
            return False

        if not self.image_files:
            messagebox.showwarning("Warning", "No images to save.")
            return False

        image_file = self.image_files[self.current_index]
        mask_file = image_file.replace('.jpg', '.png')
        self.saved_files[selected_class].append((image_file, mask_file))

        print(f"Saved files for {selected_class}:", self.saved_files[selected_class])
        return True

    def delete_pair(self) -> None:
        """
        Delete the current image and its corresponding mask.
        """
        if not self.image_files:
            return

        image_file = self.image_files[self.current_index]
        mask_file = image_file.replace('.jpg', '.png')

        # Remove the pair from the saved_files if it exists
        for class_files in self.saved_files.values():
            if (image_file, mask_file) in class_files:
                class_files.remove((image_file, mask_file))

        os.remove(image_file)
        os.remove(mask_file)

        del self.image_files[self.current_index]
        del self.image_ids[image_file]

        if self.current_index >= len(self.image_files):
            self.current_index = max(0, len(self.image_files) - 1)

        self.show_image()

    def prev_image(self) -> None:
        """
        Display the previous image and its corresponding mask.
        Save the current pair if a class is selected.
        """
        if self.save_current_pair() and self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def next_image(self) -> None:
        """
        Display the next image and its corresponding mask.
        Save the current pair if a class is selected.
        """
        if self.save_current_pair() and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()

    def remove_duplicates(self) -> None:
        """
        Remove duplicate entries from the saved_files dictionary.
        """
        for class_name, files in self.saved_files.items():
            unique_files = list(set(files))  # Remove duplicates
            self.saved_files[class_name] = unique_files
            print(f"Updated files for {class_name}:", self.saved_files[class_name])

        messagebox.showinfo("Info", "Duplicate entries have been removed.")

    def save_and_exit(self) -> None:
        """
        Save changes, remove duplicate entries, move files to target directory, and exit the application.
        """
        self.remove_duplicates()

        if not self.target_directory:
            messagebox.showerror("Error", "No target directory specified.")
            return

        for class_name, files in self.saved_files.items():
            class_dir = os.path.join(self.target_directory, class_name)
            images_dir = os.path.join(class_dir, "images")
            masks_dir = os.path.join(class_dir, "masks")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            for image_file, mask_file in files:
                shutil.move(image_file, os.path.join(images_dir, os.path.basename(image_file)))
                shutil.move(mask_file, os.path.join(masks_dir, os.path.basename(mask_file)))

        self.root.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image and Mask Viewer")
    parser.add_argument(
        "--config",
        type=str,
        default="dinocut_config.yaml",
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to the initial directory to load images and masks from."
    )
    parser.add_argument(
        "--target-directory",
        type=str,
        help="Path to the target directory to move images and masks to after classification."
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = ImageMaskViewer(root, args.config, args.directory, args.target_directory)
    root.mainloop()

'''
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import yaml
import argparse
from typing import List, Dict, Any, Tuple


class ImageMaskViewer:
    def __init__(self, root: tk.Tk, config_file: str, initial_directory: str = None) -> None:
        """
        Initialize the ImageMaskViewer application.

        Args:
            root (tk.Tk): The root window of the Tkinter application.
            config_file (str): Path to the configuration YAML file.
            initial_directory (str, optional): The initial directory to load images and masks from.
        """
        self.root = root
        self.root.title("Image and Mask Viewer")
        self.directory = initial_directory or ""
        self.image_files: List[str] = []
        self.image_ids: Dict[str, int] = {}
        self.current_index = 0
        self.config_file = config_file
        self.class_labels: List[str] = self.load_class_labels()
        self.selected_class = tk.StringVar()
        self.saved_files: Dict[str, List[Tuple[str, str]]] = {label: [] for label in self.class_labels}

        self.load_button = tk.Button(self.root, text="Load Directory", command=self.load_directory)
        self.load_button.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side="left", padx=10)

        self.mask_label = tk.Label(self.root)
        self.mask_label.pack(side="right", padx=10)

        self.id_label = tk.Label(self.root, text="Image ID:")
        self.id_label.pack()

        self.id_text = tk.Entry(self.root, state='readonly')
        self.id_text.pack()

        self.radio_frame = tk.Frame(self.root)
        self.radio_frame.pack(pady=10)

        for label in self.class_labels:
            rb = tk.Radiobutton(self.radio_frame, text=label, variable=self.selected_class, value=label)
            rb.pack(anchor="w")

        self.delete_button = tk.Button(self.root, text="Delete Pair", command=self.delete_pair)
        self.delete_button.pack()

        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side="left", padx=10)

        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(side="right", padx=10)

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side="bottom", fill="x")

        self.exit_button = tk.Button(self.bottom_frame, text="Save and Exit", command=self.save_and_exit)
        self.exit_button.pack(pady=20)

        if initial_directory:
            self.load_images_from_directory(initial_directory)

    def load_class_labels(self) -> List[str]:
        """
        Load class labels from the configuration YAML file.

        Returns:
            List[str]: A list of class labels.
        """
        with open(self.config_file, "r") as file:
            config: Dict[str, Any] = yaml.safe_load(file)
            return config.get("image_settings", {}).get("classes", [])

    def load_directory(self) -> None:
        """
        Open a file dialog to select a directory and load all .jpg image files and corresponding .png mask files from it and its subdirectories.
        """
        self.directory = filedialog.askdirectory()
        if not self.directory:
            return
        self.load_images_from_directory(self.directory)

    def load_images_from_directory(self, directory: str) -> None:
        """
        Load all .jpg image files and corresponding .png mask files from the specified directory and its subdirectories.

        Args:
            directory (str): The directory to load images and masks from.
        """
        self.image_files = []
        self.image_ids = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(root, file)
                    mask_path = image_path.replace('.jpg', '.png')
                    if os.path.exists(mask_path):
                        self.image_files.append(image_path)
        self.image_ids = {f: idx + 1 for idx, f in enumerate(self.image_files)}  # Assign a sequential ID to each image file
        self.current_index = 0
        self.show_image()

    def show_image(self) -> None:
        """
        Display the current image and its corresponding mask side by side.
        """
        if not self.image_files:
            return

        image_file = self.image_files[self.current_index]
        mask_file = image_file.replace('.jpg', '.png')

        if not os.path.exists(mask_file):
            messagebox.showerror("Error", f"Mask file {mask_file} not found")
            return

        image = Image.open(image_file)
        mask = Image.open(mask_file)

        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        mask = mask.resize((300, 300), Image.Resampling.LANCZOS)

        self.image_tk = ImageTk.PhotoImage(image)
        self.mask_tk = ImageTk.PhotoImage(mask)

        self.image_label.config(image=self.image_tk)
        self.mask_label.config(image=self.mask_tk)

        # Display the unique ID of the current image
        image_id = self.image_ids[image_file]
        self.id_text.config(state='normal')
        self.id_text.delete(0, tk.END)
        self.id_text.insert(0, str(image_id))
        self.id_text.config(state='readonly')

    def save_current_pair(self) -> bool:
        """
        Save the current image and its corresponding mask to the selected class list.
        If no class is selected, show a warning message.

        Returns:
            bool: True if the pair is saved successfully, False otherwise.
        """
        selected_class = self.selected_class.get()
        if not selected_class:
            messagebox.showwarning("Warning", "Please select a class before proceeding.")
            return False

        if not self.image_files:
            messagebox.showwarning("Warning", "No images to save.")
            return False

        image_file = self.image_files[self.current_index]
        mask_file = image_file.replace('.jpg', '.png')
        self.saved_files[selected_class].append((image_file, mask_file))

        print(f"Saved files for {selected_class}:", self.saved_files[selected_class])
        return True

    def delete_pair(self) -> None:
        """
        Delete the current image and its corresponding mask.
        """
        if not self.image_files:
            return

        image_file = self.image_files[self.current_index]
        mask_file = image_file.replace('.jpg', '.png')

        # Remove the pair from the saved_files if it exists
        for class_files in self.saved_files.values():
            if (image_file, mask_file) in class_files:
                class_files.remove((image_file, mask_file))

        os.remove(image_file)
        os.remove(mask_file)

        del self.image_files[self.current_index]
        del self.image_ids[image_file]

        if self.current_index >= len(self.image_files):
            self.current_index = max(0, len(self.image_files) - 1)

        self.show_image()

    def prev_image(self) -> None:
        """
        Display the previous image and its corresponding mask.
        Save the current pair if a class is selected.
        """
        if self.save_current_pair() and self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def next_image(self) -> None:
        """
        Display the next image and its corresponding mask.
        Save the current pair if a class is selected.
        """
        if self.save_current_pair() and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()

    def remove_duplicates(self) -> None:
        """
        Remove duplicate entries from the saved_files dictionary.
        """
        for class_name, files in self.saved_files.items():
            unique_files = list(set(files))  # Remove duplicates
            self.saved_files[class_name] = unique_files
            print(f"Updated files for {class_name}:", self.saved_files[class_name])

        messagebox.showinfo("Info", "Duplicate entries have been removed.")

    def save_and_exit(self) -> None:
        """
        Save changes, remove duplicate entries, and exit the application.
        """
        self.remove_duplicates()
        self.root.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image and Mask Viewer")
    parser.add_argument(
        "--config",
        type=str,
        default="dinocut_config.yaml",
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to the initial directory to load images and masks from."
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = ImageMaskViewer(root, args.config, args.directory)
    root.mainloop()
'''
