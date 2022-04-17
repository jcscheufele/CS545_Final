from collections import OrderedDict
from torch import as_tensor, nn, no_grad, cat, flatten, float32, uint8, div, tensor, mean, reshape, unsqueeze
from torchvision import transforms
import numpy as np
import wandb
from random import sample
import cv2 as cv

class BasicNetwork(nn.Module):
  def __init__(self, in_features, out_features):
    super(BasicNetwork,self).__init__()
    self.convolution = nn.Sequential(OrderedDict([
        ("Input", nn.Conv2d(3, 8, kernel_size=40, padding=1)),
        ("ReLU 1", nn.ReLU()),
        ("Max Pooling 2", nn.MaxPool2d(4, 4)),
        ("Hidden 1", nn.Conv2d(8, 32, kernel_size=40, padding=1)),
        ("ReLU 2", nn.ReLU()),
        ("Max Pooling 1", nn.MaxPool2d(2, 2)),

        ("Hidden 2", nn.Conv2d(32, 64, kernel_size=10, padding=1)),
        ("ReLU 3", nn.ReLU()),
        ("Max Pooling 2", nn.MaxPool2d(2, 2)),
        ("Hidden 3", nn.Conv2d(64, 128, kernel_size=10, padding=1)),
        ("ReLU 4", nn.ReLU()),
        ("Max Pooling 2", nn.MaxPool2d(2, 2)),
        ("Output Flatten", nn.Flatten())
    ]))
    self.longer_stack = nn.Sequential(OrderedDict([ #Creating an ordered dictionary of the 3 layers we want in our NN
        ('Input', nn.Linear(133632, 4096)),
        ('Relu 1', nn.ReLU()),
        ('Hidden Linear 1', nn.Linear(4096, 1024)),
        ('Relu 2', nn.ReLU()),
        ('Hidden Linear 2', nn.Linear(1024, 256)),
        ('Relu 3', nn.ReLU()),
        ('Hidden Linear 3', nn.Linear(256, 128)),
        ('Relu 4', nn.ReLU()),
        ('Hidden Linear 4', nn.Linear(128, 64)),
        ('Relu 5', nn.ReLU()),
        ('Hidden Linear 5', nn.Linear(64, 32)),
        ('Relu 6', nn.ReLU()),
        ('Hidden Linear 6', nn.Linear(32, 16)),
        ('Relu 7', nn.ReLU()),
        ('Hidden Linear 7', nn.Linear(16, 8)),
        ('Relu 8', nn.ReLU()),
        ('Output', nn.Linear(8, out_features))
    ]))
    self.wider_stack = nn.Sequential(OrderedDict([ #Creating an ordered dictionary of the 3 layers we want in our NN
        ('Input', nn.Linear(409600, 8192)),
        ('Relu 1', nn.ReLU()),
        ('Hidden Linear 1', nn.Linear(8192, 4096)),
        ('Relu 2', nn.ReLU()),
        ('Hidden Linear 2', nn.Linear(4096, 1024)),
        ('Relu 3', nn.ReLU()),
        ('Hidden Linear 3', nn.Linear(1024, 16)),
        ('Relu 4', nn.ReLU()),
        ('Output', nn.Linear(8, out_features))
    ]))
    self.simple_stack = nn.Sequential(OrderedDict([ #Creating an ordered dictionary of the 3 layers we want in our NN
        ('Input', nn.Linear(409600, 1024)),
        ('Relu 1', nn.ReLU()),
        ('Hidden Linear 1', nn.Linear(1024, 16)),
        ('Relu 2', nn.ReLU()),
        ('Output', nn.Linear(16, out_features))
    ]))

  #Defining how the data flows through the layer, PyTorch will call this we will never have to call it
  def forward(self, x):   # (62, 128, 440*426+3) - > (62, 128, 240, 426), (62, 128, 240, 426)
    logits = self.convolution(x)
    logits = self.longer_stack(logits)
    return logits

def recOverlap(lTx, rTx, lPx, rPx, lTy, rTy, lPy, rPy):
     
    # To check if either rectangle is actually a line
      # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}

    #print(lT[0].item())
       
    if ((lTx == rTx) or (lTy == rTy) or (lPx == rPx) or (lPy == rPy)):
        # the line cannot have positive overlap
        print(1)
        return False
       
    # If one rectangle is on left side of other
    if((lTx >= rPx) or (lPx >= rTx)):
        print(2)
        return False
 
    # If one rectangle is above other
    if((rTy >= lPy) or (rPy >= lTy)):
        print(3)
        return False

    return True


def overlappingArea(lTx, rTx, lPx, rPx, lTy, rTy, lPy, rPy):
    x = 0
    y = 1
 
    # Area of 1st Rectangle
    area1 = abs(lTx - rTx) * abs(lTy - rTy)
 
    # Area of 2nd Rectangle
    area2 = abs(lPx - rPx) * abs(lPy - rPy)
 
    ''' Length of intersecting part i.e 
        start from max(l1[x], l2[x]) of 
        x-coordinate and end at min(r1[x],
        r2[x]) x-coordinate by subtracting 
        start from end we get required 
        lengths '''
    x_dist = (min(rTx, rPx) -
              max(lTx, lPx))
 
    y_dist = (min(rTy, rPy) -
              max(lTy, lPy))
    areaI = 0
    if x_dist > 0 and y_dist > 0:
        areaI = x_dist * y_dist
    else:
        areaI = 0

    if area1 >= area2:
        ratio = area1/area2
    else:
        ratio = area2/area1

    coverage = 1 - areaI/area1
 
    return coverage*ratio

def percent_error(lT, rT, lP, rP):
    error = []
    #print(lT)
    for i in range(len(lT)):
        #print(lT[i])
        lTx, rTx, lPx, rPx, lTy, rTy, lPy, rPy = int(lT[i][0].item()), int(rT[i][0].item()), int(lP[i][0].item()), int(rP[i][0].item()), int(lT[i][1].item()), int(rT[i][1].item()), int(lP[i][1].item()), int(rP[i][1].item())
        if recOverlap(lTx, rTx, lPx, rPx, lTy, rTy, lPy, rPy):
            single_err = overlappingArea(lTx, rTx, lPx, rPx, lTy, rTy, lPy, rPy)
            print("Error:", single_err)
            error.append(single_err)
        else:
            error.append(10000)
    return error

# Model Evaluation #############################################################
#Takes in a dataloader, a NN model, our loss function, and an optimizer and trains the NN 
def train_loop(dataloader, model, loss_fn, optimizer, device, epoch, bs, will_save, key):
    batches = int(len(dataloader.dataset)*.8/bs)
    cumulative_loss = 0
    cumulative_error = 0
    ret = []

    for batch, (X, y) in enumerate(dataloader):
        y=y.to(device)
        pred = model(X.to(device))
        loss = loss_fn(pred, y)

        #y = y.cpu().numpy()
        #pred = pred.cpu().numpy()

        error = percent_error(y[:,0:2], y[:, 2:4], pred[:, 0:2], pred[:, 2:4])

        cumulative_loss += loss
        cumulative_error += np.mean(error)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if will_save: # and (batch % 10 == 0):
            range = sample(list(np.arange(len(X))), min(len(X), 10))
            for idx in range:
                #print(X[idx].size())
                img = X[idx].detach().cpu().numpy()
                img = np.reshape(img, (240, 400, 3))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                #img = cv.cvtColor(X[idx].detach().cpu().numpy(), cv.COLOR_GRAY2BGR)
                img = cv.rectangle(img, [int(y[idx, 1]), int(y[idx, 0])], [int(y[idx, 3]), int(y[idx, 2])], color=(255, 0, 0))
                img = cv.rectangle(img, [int(pred[idx, 1]), int(pred[idx, 0])], [int(pred[idx, 3]), int(pred[idx, 2])], color=(0,255,0))
                save = {"Train Key": key, "Sample Epoch":epoch,
                "Sample Training Loss":loss,
                "Sample Training Percent Error" : error[idx],
                "Sample Training Coposite": wandb.Image(img)}
                ret.append(save)
                key+=1
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss} SAVED\n"
            print(row) 
        else:
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss}\n"
            print(row)

    averages_1 = f"End of Training \n Test Error: \n Training Avg loss: {(cumulative_loss/batches):>8f}\n"
    print(averages_1)
    return ret, cumulative_loss/batches, cumulative_error/batches, key


def test_loop(dataloader, model, loss_fn, device, epoch, bs, will_save, key):
    batches = int(len(dataloader.dataset)*.8/bs)
    cumulative_loss = 0
    cumulative_error = 0
    ret = []
    with no_grad():
        for batch, (X, y) in enumerate(dataloader):
            y=y.to(device)
            pred = model(X.to(device))
            loss = loss_fn(pred, y)

            #y = y.cpu().numpy()
            #pred = pred.cpu().numpy()

            error = percent_error(y[:,0:2], y[:, 2:4], pred[:, 0:2], pred[:, 2:4])

            cumulative_loss += loss
            cumulative_error += np.mean(error)

            if will_save: # and (batch % 10 == 0):
                range = sample(list(np.arange(len(X))), min(len(X), 10))
                for idx in range:
                    #print(X[idx].size())
                    img = X[idx].detach().cpu().numpy()
                    img = np.reshape(img, (240, 400, 3))
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    #img = cv.cvtColor(X[idx].detach().cpu().numpy(), cv.COLOR_GRAY2BGR)
                    img = cv.rectangle(img, [int(y[idx, 1]), int(y[idx, 0])], [int(y[idx, 3]), int(y[idx, 2])], color=(255, 0, 0))
                    img = cv.rectangle(img, [int(pred[idx, 1]), int(pred[idx, 0])], [int(pred[idx, 3]), int(pred[idx, 2])], color=(0,255,0))
                    save = {"Test Key": key, "Sample Epoch":epoch,
                    "Sample Testing Loss":loss,
                    "Sample Testing Percent Error" : error[idx],
                    "Sample Testing Coposite": wandb.Image(img)}
                    ret.append(save)
                    key+=1
                row = f"{epoch} [ {batch}/{batches} ] Loss: {loss} SAVED\n"
                print(row) 
            else:
                row = f"{epoch} [ {batch}/{batches} ] Loss: {loss}\n"
                print(row)

    averages_1 = f"End of Testing \n Test Error: \n Testing Avg loss: {(cumulative_loss/batches):>8f}\n"
    print(averages_1)
    return ret, cumulative_loss/batches, cumulative_error/batches, key
