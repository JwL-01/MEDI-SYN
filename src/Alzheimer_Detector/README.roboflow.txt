
Alzheimer Detector - v1 2023-06-14 5:43pm
==============================

This dataset was exported via roboflow.com on October 26, 2023 at 12:46 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2006 images.
Alzheimer are annotated in Tensorflow Object Detection format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Grayscale (CRT phosphor)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 3 versions of each source image:
* Random exposure adjustment of between -25 and +25 percent
* Salt and pepper noise was applied to 5 percent of pixels


