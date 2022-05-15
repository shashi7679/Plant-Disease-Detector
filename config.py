from pickle import TRUE
import os
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
IMAGE_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = TRUE
DATASET_PATH = "./crowdai/"
SAVED_PATH = "./saved/Models"
VAL_SIZE = 0.2
MODEL = "ResNet18"
N_CLASSES = len(os.listdir(DATASET_PATH))