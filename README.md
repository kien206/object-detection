# Object Detection

2023.1 Introduction to Deep Learning Capstone Project

This repository contains the training process for each model in the project.

## Dataset
COCO minitrain is a subset of the original COCO dataset, and contains 25K images (about 20\% of the original set) and around 184K annotations across 80 object categories. We randomly sampled these images from the full set while preserving the following three quantities as much as possible: proportion of object instances from each class; overall ratios of small, medium and large objects; per class ratios of small, medium and large objects.

Link to the dataset: https://www.kaggle.com/datasets/quynhongt/data-deep

## Training process
Each folder has 2 files: the first one is the model which we train from scracth with our dataset, the second one is the model with pretrained weights.
If you want to train the model from scratch, you need to run the no-pretrained notebook with our dataset.
If you want to evaluate the pretrained model, run the pretrained notebook.

You can find our trained weights for each model here: https://drive.google.com/drive/folders/1Zt5VBdgbmD3NDtrys33jsGzeOkBKb1tZ
