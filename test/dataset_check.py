import os
import unittest
import json

class TestDatasetConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Replace with the actual path to your dataset
        cls.dataset_path = '/home/nalkuq/cmm/dataset'
    
    def get_files(self, directory, extensions):
        """
        Returns a list of files in the given directory with the specified extensions.
        """
        if not os.path.exists(directory):
            self.fail(f"Directory does not exist: {directory}")
        return [f for f in os.listdir(directory) if any(f.endswith(ext) for ext in extensions)]
    
    def test_image_label_consistency(self):
        """
        Checks that each subdirectory (train, test, val) within the dataset directory has the same number
        of images (.jpg files) and labels (.txt or .xml files), except for COCO-style annotations.
        """
        subdirectories = ['train', 'test', 'val']

        for subdirectory in subdirectories:
            images_path = os.path.join(self.dataset_path, subdirectory, 'images')
            labels_path = os.path.join(self.dataset_path, subdirectory, 'labels')

            images = self.get_files(images_path, ['.jpg'])
            json_files = self.get_files(labels_path, ['.json'])
            
            # Skip COCO-style annotations
            if json_files:
                continue

            labels = self.get_files(labels_path, ['.txt', '.xml'])

            num_images = len(images)
            num_labels = len(labels)

            self.assertEqual(num_images, num_labels,
                f"Discrepancy found in the '{subdirectory}' subdirectory: "
                f"{num_images} images and {num_labels} labels.")
    
    def test_image_label_filenames(self):
        """
        Checks that each .jpg file has a corresponding .txt or .xml file with the same name in each subdirectory,
        except for COCO-style annotations.
        """
        subdirectories = ['train', 'test', 'val']

        for subdirectory in subdirectories:
            images_path = os.path.join(self.dataset_path, subdirectory, 'images')
            labels_path = os.path.join(self.dataset_path, subdirectory, 'labels')

            image_files = self.get_files(images_path, ['.jpg'])
            json_files = self.get_files(labels_path, ['.json'])
            
            # Skip COCO-style annotations
            if json_files:
                continue

            label_files = self.get_files(labels_path, ['.txt', '.xml'])

            image_names = {os.path.splitext(f)[0] for f in image_files}
            label_names = {os.path.splitext(f)[0] for f in label_files}

            self.assertSetEqual(image_names, label_names,
                f"Filename discrepancy found in the '{subdirectory}' subdirectory: "
                f"Images without labels: {image_names - label_names}, "
                f"Labels without images: {label_names - image_names}")
    
    def test_label_format(self):
        """
        Checks that the labels in each subdirectory are either all .txt (YOLO format), all .xml (VOC format),
        or a single .json (COCO format).
        """
        subdirectories = ['train', 'test', 'val']

        for subdirectory in subdirectories:
            labels_path = os.path.join(self.dataset_path, subdirectory, 'labels')
            label_files = self.get_files(labels_path, ['.txt', '.xml', '.json'])

            yolo_labels = [f for f in label_files if f.endswith('.txt')]
            voc_labels = [f for f in label_files if f.endswith('.xml')]
            coco_labels = [f for f in label_files if f.endswith('.json')]

            # Ensure that either all labels are .txt, all are .xml, or there is one .json file
            self.assertTrue(
                (len(yolo_labels) > 0 and len(voc_labels) == 0 and len(coco_labels) == 0) or
                (len(voc_labels) > 0 and len(yolo_labels) == 0 and len(coco_labels) == 0) or
                (len(coco_labels) == 1 and len(yolo_labels) == 0 and len(voc_labels) == 0),
                f"Mixed or incorrect label formats found in the '{subdirectory}' subdirectory: "
                f"{len(yolo_labels)} .txt files, {len(voc_labels)} .xml files, and {len(coco_labels)} .json files."
            )
    
    def test_coco_annotations(self):
        """
        Checks that COCO .json annotation files are properly structured.
        """
        subdirectories = ['train', 'test', 'val']

        for subdirectory in subdirectories:
            labels_path = os.path.join(self.dataset_path, subdirectory, 'labels')
            json_files = self.get_files(labels_path, ['.json'])

            for json_file in json_files:
                with open(os.path.join(labels_path, json_file), 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        self.fail(f"Invalid JSON format in file: {json_file}")

                    # Check for necessary keys in COCO format
                    required_keys = {'images', 'annotations', 'categories'}
                    self.assertTrue(required_keys.issubset(data.keys()),
                        f"Missing required keys in COCO annotation file: {json_file}")

if __name__ == '__main__':
    unittest.main()
