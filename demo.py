import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the Image Viewer application.

        Args:
            root (tk.Tk): The root window of the Tkinter application.
        """
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Select a directory to view images", font=("Arial", 14))
        self.label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=700, height=500)
        self.canvas.pack()

        self.dir_button = tk.Button(root, text="Select Directory", command=self.select_directory)
        self.dir_button.pack(side=tk.LEFT, padx=20, pady=20)

        self.next_button = tk.Button(root, text="Next Image", command=self.next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=20)

        self.prev_button = tk.Button(root, text="Previous Image", command=self.prev_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.RIGHT, padx=10, pady=20)

        self.images: list[str] = []
        self.current_image_index: int = -1

    def select_directory(self) -> None:
        """
        Open a file dialog to select a directory and load all image files.
        """
        directory = filedialog.askdirectory(title="Select a directory")
        if directory:
            self.images = [os.path.join(directory, file) for file in os.listdir(directory) 
                           if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            self.current_image_index = 0
            if self.images:
                self.display_image()
                self.next_button.config(state=tk.NORMAL if len(self.images) > 1 else tk.DISABLED)
                self.prev_button.config(state=tk.NORMAL if len(self.images) > 1 else tk.DISABLED)
            else:
                messagebox.showinfo("No Images", "No images found in the selected directory.")
                self.canvas.delete("all")
                self.label.config(text="Select a directory to view images")
                self.next_button.config(state=tk.DISABLED)
                self.prev_button.config(state=tk.DISABLED)

    def display_image(self) -> None:
        """
        Display the current image on the canvas.
        """
        if 0 <= self.current_image_index < len(self.images):
            image_path = self.images[self.current_image_index]
            image = Image.open(image_path)
            image = image.resize((700, 500), Image.Resampling.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(image)
            self.canvas.create_image(350, 250, image=self.image_tk)
            self.label.config(text=os.path.basename(image_path))

    def next_image(self) -> None:
        """
        Display the next image in the directory.
        """
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.display_image()

    def prev_image(self) -> None:
        """
        Display the previous image in the directory.
        """
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()
