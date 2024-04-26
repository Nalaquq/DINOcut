from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import torch
import supervision as sv

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
device = "cuda"

SAM_ENCODER_VERSION = "vit_h"
sam = sam_model_registry[SAM_ENCODER_VERSION](
    checkpoint="sam/weights/sam_vit_h_4b8939.pth"
)
sam.to(device)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)
img = cv2.imread("examples/images/spongebob caveman meme.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(img)
print(masks)


"""
detections = sv.Detections.from_sam(masks)
box_annotator = sv.BoundingBoxAnnotator()
labels = [
	f"{classes[class_id]} {confidence:0.2f}"
	for _, _, confidence, class_id, _
	in detections
]

annotated_frame = box_annotator.annotate(
	scene=image.copy(),
	detections=detections,
	labels=labels
)

sv.plot_image(image=annotated_frame, size=(16, 16))



for mask in mask: 
    cv2.imshow('mask', mask)
cv2.imshow('img', img)
"""
