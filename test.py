import VisualFlow as vf

vf.to_coco(in_format='yolo',
       images='path/to/images',
       annotations='path/to/annotations',
       class_file='path/to/classes.txt',
       output_file_path='path/to/output.json')