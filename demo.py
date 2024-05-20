import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageMaskViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and Mask Viewer")
        self.directory = ""
        self.image_files = []
        self.current_index = 0

        self.load_button = tk.Button(self.root, text="Load Directory", command=self.load_directory)
        self.load_button.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side="left", padx=10)

        self.mask_label = tk.Label(self.root)
        self.mask_label.pack(side="right", padx=10)

        self.delete_button = tk.Button(self.root, text="Delete Pair", command=self.delete_pair)
        self.delete_button.pack()

        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side="left", padx=10)

        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(side="right", padx=10)

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side="bottom", fill="x")

        self.save_exit_button = tk.Button(self.bottom_frame, text="Save and Exit", command=self.save_and_exit)
        self.save_exit_button.pack(pady=20)

    def load_directory(self):
        self.directory = filedialog.askdirectory()
        if not self.directory:
            return
        self.image_files = [f for f in os.listdir(self.directory) if f.endswith('.jpg')]
        self.current_index = 0
        self.show_image()

    def show_image(self):
        if not self.image_files:
            return

        image_file = self.image_files[self.current_index]
        mask_file = image_file.replace('.jpg', '.png')

        image_path = os.path.join(self.directory, image_file)
        mask_path = os.path.join(self.directory, mask_file)

        if not os.path.exists(mask_path):
            messagebox.showerror("Error", f"Mask file {mask_file} not found")
            return

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        mask = mask.resize((300, 300), Image.Resampling.LANCZOS)

        self.image_tk = ImageTk.PhotoImage(image)
        self.mask_tk = ImageTk.PhotoImage(mask)

        self.image_label.config(image=self.image_tk)
        self.mask_label.config(image=self.mask_tk)

    def delete_pair(self):
        if not self.image_files:
            return

        image_file = self.image_files[self.current_index]
        mask_file = image_file.replace('.jpg', '.png')

        image_path = os.path.join(self.directory, image_file)
        mask_path = os.path.join(self.directory, mask_file)

        os.remove(image_path)
        os.remove(mask_path)

        del self.image_files[self.current_index]

        if self.current_index >= len(self.image_files):
            self.current_index = max(0, len(self.image_files) - 1)

        self.show_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()

    def save_and_exit(self):
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageMaskViewer(root)
    root.mainloop()

