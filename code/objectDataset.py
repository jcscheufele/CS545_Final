import numpy as np
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torch import as_tensor, div, flatten, cat, unsqueeze, reshape
import torch
import cv2 as cv
import pandas as pd
import urllib3
import pickle

class BasicDataset(Dataset):
    def __init__(self, start, end):
        self.X = []
        self.y = []
        #self.makeNewFeature()
        self.__makefeatures__(start, end)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __makefeatures__(self, start, end):
        print("Making Features...")
        boxes = pickle.load(open("annotations.pkl", "rb"))
        for idx in range(start, end):
            print(f"[ {idx}/{self.__len__()} ]", end='\r')
            car = cv.imread(f"../Images/{idx:04}.jpg")
            car = cv.resize(car, (426, 240), interpolation = cv.INTER_AREA)
            uL = boxes[idx][0]
            lR = boxes[idx][1]

            self.X.append(reshape(as_tensor(car, dtype=torch.float32), (3, 240, 426)))
            self.y.append(as_tensor([uL[0], uL[1], lR[0], lR[1]], dtype=torch.float32))
            #break
            #exx.append(torch.unsqueeze(input, dim=0)) #flattens each image and concatonates them and appends them to a list
        print("Features Created")