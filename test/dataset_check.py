import os
import unittest

class TestDatasetConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Replace with the actual path to your dataset
        cls.dataset_path = '/home/nalkuq/cmm/dataset'
    
    def get_files(self, directory, extension):
        """
        Returns a list of files in the given directory with the specified extension.
        """
        if not os.path.exists(directory):
            self.fail(f"Directory does not exist: {directory}")
        return [f for f in os.listdir(directory) if f.endswith(extension)]
    
    def test_image_label_consistency(self):
        """
        Checks that each subdirectory (train, test, val) within the dataset directory has the same number
        of images (.jpg files) and labels (.txt files).
        """
        subdirectories = ['train', 'test', 'val']

        for subdirectory in subdirectories:
            images_path = os.path.join(self.dataset_path, subdirectory, 'images')
            labels_path = os.path.join(self.dataset_path, subdirectory, 'labels')

            images = self.get_files(images_path, '.jpg')
            labels = self.get_files(labels_path, '.txt')

            num_images = len(images)
            num_labels = len(labels)

            self.assertEqual(num_images, num_labels,
                f"Discrepancy found in the '{subdirectory}' subdirectory: "
                f"{num_images} images and {num_labels} labels.")
    
    def test_image_label_filenames(self):
        """
        Checks that each .jpg file has a corresponding .txt file with the same name in each subdirectory.
        """
        subdirectories = ['train', 'test', 'val']

        for subdirectory in subdirectories:
            images_path = os.path.join(self.dataset_path, subdirectory, 'images')
            labels_path = os.path.join(self.dataset_path, subdirectory, 'labels')

            image_files = self.get_files(images_path, '.jpg')
            label_files = self.get_files(labels_path, '.txt')

            image_names = {os.path.splitext(f)[0] for f in image_files}
            label_names = {os.path.splitext(f)[0] for f in label_files}

            self.assertSetEqual(image_names, label_names,
                f"Filename discrepancy found in the '{subdirectory}' subdirectory: "
                f"Images without labels: {image_names - label_names}, "
                f"Labels without images: {label_names - image_names}")

if __name__ == '__main__':
    unittest.main()
