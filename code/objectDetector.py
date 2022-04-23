from collections import OrderedDict
from torch import as_tensor, nn, no_grad, cat, flatten, float32, uint8, div, tensor, mean, reshape, unsqueeze
from torchvision import transforms
import numpy as np
import wandb
from random import sample
import cv2 as cv

class BasicNetwork(nn.Module):
  def __init__(self, out_features):
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
        ('Input', nn.Linear(147456, 4096)),
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
        ('Input', nn.Linear(147456, 8192)),
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
        ('Input', nn.Linear(147456, 1024)),
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


def overlapArea(A_x, A_y, B_x, B_y, a_x, a_y, b_x, b_y):
    X = 0
    Y = 0
    if not((a_x > B_x) or (A_x>b_x)):
        X = min(B_x, b_x) - max(A_x, a_x)
    if not((a_y > B_y) or (A_y > b_y)):
        Y = min(B_y, b_y) - max(A_y, a_y)
    return X*Y


def percent_error(uL_truth, lR_truth, uL_pred, lR_pred):
    error = []
    for i in range(len(uL_truth)):
        #print(uL_truth[i])

        uL_truth_x = int(uL_truth[i][0].item())
        uL_truth_y = int(uL_truth[i][1].item())
        lR_truth_x = int(lR_truth[i][0].item())
        lR_truth_y = int(lR_truth[i][1].item())

        uL_pred_x = int(uL_pred[i][0].item())
        uL_pred_y = int(uL_pred[i][1].item())
        lR_pred_x = int(lR_pred[i][0].item())
        lR_pred_y = int(lR_pred[i][1].item())

        intersect_area = overlapArea(
            uL_truth_x, uL_truth_y, lR_truth_x, lR_truth_y,
            uL_pred_x, uL_pred_y, lR_pred_x, lR_pred_y
        )
        
        width_truth = lR_truth_x - uL_truth_x
        height_truth = lR_truth_y - uL_truth_y
        area_truth = width_truth * height_truth

        coverage = 1 - intersect_area/area_truth
    
        error.append(coverage)
        
    return error

# Model Evaluation #############################################################
#Takes in a dataloader, a NN model, our loss function, and an optimizer and trains the NN 
def train_loop(dataloader, model, loss_fn, optimizer, device, epoch, bs, will_save, key):
    batches = int(len(dataloader.dataset)/bs)
    cumulative_loss = 0
    cumulative_error = 0
    ret = []

    for batch, (X, y) in enumerate(dataloader):
        y=y.to(device)
        pred = model(X.to(device))
        loss = loss_fn(pred, y)

        #y = y.cpu().numpy()
        #pred = pred.cpu().numpy()
        
        uL_truth = y[:,0:2]
        lR_truth = y[:, 2:4]

        uL_pred = pred[:, 0:2]
        lR_pred = pred[:, 2:4]

        
        error = percent_error(uL_truth, lR_truth, uL_pred, lR_pred)

        cumulative_loss += loss
        cumulative_error += np.mean(error)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if will_save and (batch % 5 == 0):
            range = sample(list(np.arange(len(X))), min(len(X), 10))
            for idx in range:
                #print(X[idx].size())
                img = X[idx].detach().cpu().numpy()
                img = np.reshape(img, (240, 426, 3))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                #img = cv.cvtColor(X[idx].detach().cpu().numpy(), cv.COLOR_GRAY2BGR)
                img = cv.rectangle(img, [int(y[idx, 0]), int(y[idx, 1])], [int(y[idx, 2]), int(y[idx, 3])], color=(255, 0, 0))
                img = cv.rectangle(img, [int(pred[idx, 0]), int(pred[idx, 1])], [int(pred[idx, 2]), int(pred[idx, 3])], color=(0,255,0))
                save = {"Train Key": key, "Sample Epoch":epoch,
                "Sample Training Loss": loss,
                "Sample Training Co-ord Truth": [int(y[idx, 0]), int(y[idx, 1]), int(y[idx, 2]), int(y[idx, 3])],
                "Sample Training Co-ord Truth": [int(pred[idx, 0]), int(pred[idx, 1]), int(pred[idx, 2]), int(pred[idx, 3])],
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
    batches = int(len(dataloader.dataset)/bs)
    cumulative_loss = 0
    cumulative_error = 0
    ret = []
    with no_grad():
        for batch, (X, y) in enumerate(dataloader):
            y=y.to(device)
            pred = model(X.to(device))
            loss = loss_fn(pred, y)

            uL_truth = y[:,0:2]
            lR_truth = y[:, 2:4]

            uL_pred = pred[:, 0:2]
            lR_pred = pred[:, 2:4]

            
            error = percent_error(uL_truth, lR_truth, uL_pred, lR_pred)

            cumulative_loss += loss
            cumulative_error += np.mean(error)

            if will_save: # and (batch % 10 == 0):
                range = sample(list(np.arange(len(X))), min(len(X), 10))
                for idx in range:
                    #print(X[idx].size())
                    img = X[idx].detach().cpu().numpy()
                    img = np.reshape(img, (240, 426, 3))
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    #img = cv.cvtColor(X[idx].detach().cpu().numpy(), cv.COLOR_GRAY2BGR)
                    img = cv.rectangle(img, [int(y[idx, 0]), int(y[idx, 1])], [int(y[idx, 2]), int(y[idx, 3])], color=(255, 0, 0))
                    img = cv.rectangle(img, [int(pred[idx, 0]), int(pred[idx, 1])], [int(pred[idx, 2]), int(pred[idx, 3])], color=(0,255,0))
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
