import numpy as np
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torch import as_tensor, div, flatten, cat, unsqueeze, reshape
import torch
import cv2 as cv
import pandas as pd
import urllib3

# search is the string you're looking for inside the log file and path is the path to the log file on your computer.
# This function pulls out the key information from each line with they key, search string.  In our case it pulls out the altitude, latitude, and longtitude of each image location.

def createTransform():
    p144 = (144, 256)
    p240 = (240, 426)
    p360 = (360, 640)
    p480 = (480, 848)
    crop = (355, 644)
    listOfTransforms = [
        transforms.Resize(p240)
        ]
    return transforms.Compose(listOfTransforms)

def corner_det(img):
    uL = []
    lR = []
    for i in range(240):
        done = False
        for j in range(400):
            if (img[i][j] == 1):
                uL = [i, j]
                done = True
                break
        if done:
            break
    for i in range(239, -1, -1):
        done = False
        for j in range(399, -1, -1):
            if (img[i][j] == 1):
                lR = [i, j]
                done = True
                break
        if done:
            break
    return uL, lR

class BasicDataset(Dataset):
    def __init__(self, train):
        self.X = []
        self.y = []
        self.train = train
        #self.makeNewFeature()
        self.__makefeatures__()

    def __len__(self):
        return 9

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __getimage__(self, path):
        return read_image(path, mode=ImageReadMode.GRAY)

    def __makefeatures__(self):
        print("Making Features...")
        for idx in range(1, self.__len__()+1):
            print(f"[ {idx}/{self.__len__()} ]", end='\r')
            car = cv.imread(f"../Images/{idx:01}.jpg")
            target = cv.imread(f"../Images/{idx:01}-t.jpg", 0)
            car = cv.resize(car, (400, 240), interpolation = cv.INTER_AREA)
            target = cv.resize(target, (400, 240), interpolation = cv.INTER_AREA)
            #cv.imwrite("../Images/gray-t.jpg", target)
            #print(target[350])
            #print(target.shape)
            
            target[target <= 100] = 1
            target[target > 100] = 0

            uL, lR = corner_det(target)
            #print(uL, lR)
            self.X.append(reshape(as_tensor(car, dtype=torch.float32), (3, 240, 400)))
            self.y.append(as_tensor([uL[0], uL[1], lR[0], lR[1]], dtype=torch.float32))
            #break
            #exx.append(torch.unsqueeze(input, dim=0)) #flattens each image and concatonates them and appends them to a list
        print("Features Created")