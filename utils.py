import torch
import config
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler,DataLoader
import numpy as np


def Shuffler(len_dataset, val_size):
    total_indices = np.random.permutation(len_dataset)
    len_val = int(len_dataset*val_size)
    return total_indices[:len_val], total_indices[len_val:]

def get_dataset():
    data_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)),  # Ideally 256X256
                        transforms.RandomCrop((int(config.IMAGE_SIZE*0.75),int(config.IMAGE_SIZE*0.75))), # 0.75 of the original image                     # 
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(10)
    ])
    ######### Loading Images from Local ############
    dataset = torchvision.datasets.ImageFolder(root=config.DATASET_PATH,transform=data_transform)
    ######### Randomly Splitting indices into train data and val data ##########
    val_indices, train_indices = Shuffler(len(dataset),config.VAL_SIZE)
    ######### Prepraing Sampler ##############
    train_Sampler = SubsetRandomSampler(train_indices)
    val_Sampler = SubsetRandomSampler(val_indices)
    ######## Loading Dataset #################
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=train_Sampler, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=val_Sampler, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    return train_loader,val_loader


