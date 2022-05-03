from torch import nn, gt
import torch
import numpy as np
import pickle
import cv2 as cv

class CIoU(nn.Module):
    
    def __init__(self):
        super(CIoU, self).__init__()

    def forward(self, inputs, targets):
        size = len(inputs)

        uL_truth = targets[:, 0:2]
        lR_truth = targets[:, 2:4]
        uL_pred = inputs[:, 0:2]
        lR_pred = inputs[:, 2:4]
        truth_cen = torch.div(torch.add(uL_truth, lR_truth), 2)
        pred_cen = torch.div(torch.add(uL_pred, lR_pred), 2)

        uL_truth_x = uL_truth[:, 0] #Ax
        uL_truth_y = uL_truth[:, 1] #Ay

        lR_truth_x = lR_truth[:, 0] #Bx
        lR_truth_y = lR_truth[:, 1] #By

        uL_pred_x = uL_pred[:, 0] #ax
        uL_pred_y = uL_pred[:, 1] #ay

        lR_pred_x = lR_pred[:, 0] #bx
        lR_pred_y = lR_pred[:, 1] #by

        #print("Centers: ", truth_cen, pred_cen)

        truth_cen_x = truth_cen[:, 0]
        truth_cen_y = truth_cen[:, 1]
        pred_cen_x = pred_cen[:, 0]
        pred_cen_y = pred_cen[:, 1]

        p = torch.sqrt((pred_cen_x - truth_cen_x)**2 + (pred_cen_y - truth_cen_y)**2)

        #print("Center diffs", p)

        enc_X = torch.reshape(torch.minimum(uL_truth, uL_pred), (size, 2))
        enc_Y = torch.reshape(torch.maximum(lR_truth, lR_pred), (size, 2))
        bounding_box = torch.reshape(torch.cat((enc_X, enc_Y), 1), (size, 4))
        
        #print('Enclosing', enc_X, enc_Y, bounding_box)

        bb_uL_x = bounding_box[:, 0]
        bb_uL_y = bounding_box[:, 1]
        bb_lR_x = bounding_box[:, 2]
        bb_lR_y = bounding_box[:, 3]

        BBA = (bb_lR_x - bb_uL_x)*(bb_lR_y-bb_uL_y)

        C = torch.sqrt((bb_lR_x - bb_uL_x)**2 + (bb_lR_y - bb_uL_y)**2)

        #print("C: ", C)

        X = torch.where(
            (torch.gt(uL_pred_x, lR_truth_x) | torch.gt(uL_truth_x, lR_pred_x)), 
            0, 
            1
        )*(torch.minimum(lR_truth_x, lR_pred_x) - torch.maximum(uL_truth_x, uL_pred_x))

        Y = torch.where(
            (torch.gt(uL_pred_y, lR_truth_y) | torch.gt(uL_truth_y, lR_pred_y)), 
            0, 
            1
        )*(torch.minimum(lR_truth_y, lR_pred_y) - torch.maximum(uL_truth_y, uL_pred_y))

        i_area = X*Y
        rec_1 = (lR_truth_x - uL_truth_x)*(lR_truth_y - uL_truth_y)
        rec_2 = (lR_pred_x - uL_pred_x)*(lR_pred_y - uL_pred_y)
        total_area = rec_1 + rec_2 - i_area

        IoU = i_area/total_area
        DIoU = 1-IoU + (p**2)/(C**2)
        return torch.mean(DIoU)

        '''#my own calculation
        first = 1 - (i_area/BBA) # this will approach 0 when i_area == bbA
        second = torch.where(p<5, first, torch.mul(p, first))
        return torch.mean(second)'''



        '''pred_W = lR_pred_x - uL_pred_x
        pred_H = torch.where((lR_pred_y - uL_pred_y)!=0, 1, 0)*(lR_pred_y - uL_pred_y)

        truth_W = lR_truth_x - uL_truth_x
        truth_H = torch.where((lR_truth_y - uL_truth_y)!=0, 1, 0)*(lR_truth_y - uL_truth_y)

        
        V = (4/(np.pi**2))*((torch.atan(torch.div(truth_W, truth_H)) - torch.atan(torch.div(pred_W, pred_H)))**2)

        #alpha_1 = torch.div(V, ((1 - IoU)+V))
        #print("alpha 1: ", alpha_1)

        alpha = torch.where(torch.lt(IoU, 0.5), 0, 1)*torch.div(V, ((1-IoU)+V))

        #print(torch.where(torch.lt(IoU, 0.5), 0, 1))

        CIoU = 1 - (IoU - (((p**2)/(C**2)) + alpha*V))
        return CIoU'''
'''
if __name__ == "__main__":
    #loss_fn = CIoU()
    #loss_fn = nn.MSELoss(reduction='mean')
    boxes = pickle.load(open("annotations.pkl", "rb"))
    print(boxes[0], boxes[81])
    inputs = torch.reshape(torch.as_tensor(boxes[0:2]), (2, 4))
    targets = torch.reshape(torch.as_tensor(boxes[2:4]), (2, 4))
    print(inputs[0][0:2])
    print(targets)
    size = len(inputs)
    
    uL_truth = targets[:, 0:2]
    lR_truth = targets[:, 2:4]
    uL_pred = inputs[:, 0:2]
    lR_pred = inputs[:, 2:4]
    truth_cen = torch.div(torch.add(uL_truth, lR_truth), 2)
    pred_cen = torch.div(torch.add(uL_pred, lR_pred), 2)

    uL_truth_x = uL_truth[:, 0] #Ax
    uL_truth_y = uL_truth[:, 1] #Ay

    lR_truth_x = lR_truth[:, 0] #Bx
    lR_truth_y = lR_truth[:, 1] #By

    uL_pred_x = uL_pred[:, 0] #ax
    uL_pred_y = uL_pred[:, 1] #ay

    lR_pred_x = lR_pred[:, 0] #bx
    lR_pred_y = lR_pred[:, 1] #by

    print("Centers: ", truth_cen, pred_cen)

    truth_cen_x = truth_cen[:, 0]
    truth_cen_y = truth_cen[:, 1]
    pred_cen_x = pred_cen[:, 0]
    pred_cen_y = pred_cen[:, 1]

    p = torch.sqrt((pred_cen_x - truth_cen_x)**2 + (pred_cen_y - truth_cen_y)**2)

    print("Center diffs", p)

    enc_X = torch.reshape(torch.minimum(uL_truth, uL_pred), (size, 2))
    enc_Y = torch.reshape(torch.maximum(lR_truth, lR_pred), (size, 2))
    bounding_box = torch.reshape(torch.cat((enc_X, enc_Y), 1), (size, 4))
    
    print('Enclosing', enc_X, enc_Y, bounding_box)

    bb_uL_x = bounding_box[:, 0]
    bb_uL_y = bounding_box[:, 1]
    bb_lR_x = bounding_box[:, 2]
    bb_lR_y = bounding_box[:, 3]

    C = torch.sqrt((bb_lR_x - bb_uL_x)**2 + (bb_lR_y - bb_uL_y)**2)

    print("C: ", C)

    X = torch.where(
        (torch.gt(uL_pred_x, lR_truth_x) | torch.gt(uL_truth_x, lR_pred_x)), 
        0, 
        1
    )*(torch.minimum(lR_truth_x, lR_pred_x) - torch.maximum(uL_truth_x, uL_pred_x))

    Y = torch.where(
        (torch.gt(uL_pred_y, lR_truth_y) | torch.gt(uL_truth_y, lR_pred_y)), 
        0, 
        1
    )*(torch.minimum(lR_truth_y, lR_pred_y) - torch.maximum(uL_truth_y, uL_pred_y))

    i_area = X*Y
    rec_1 = (lR_truth_x - uL_truth_x)*(lR_truth_y - uL_truth_y)
    rec_2 = (lR_pred_x - uL_pred_x)*(lR_pred_y - uL_pred_y)
    total_area = rec_1 + rec_2 - i_area

    IoU = i_area/total_area
    DIoU = IoU - ((p**2)/(C**2))


    print(IoU)
    print(((p**2)/(C**2)))
    print(1-DIoU)
    print(torch.mean(1-DIoU))


    for i in range(size):
        img = np.zeros((240, 426, 3))
        cv.rectangle(img, inputs[i][0:2].numpy(), inputs[i][2:4].numpy(), color=(255, 0, 0))
        cv.rectangle(img, targets[i][0:2].numpy(), targets[i][2:4].numpy(), color=(0, 255, 0))

        cv.circle(img, (int(pred_cen_x[i].item()), int(pred_cen_y[i].item())), color=(255, 0, 0), radius=0, thickness=-1)
        cv.circle(img, (int(truth_cen_x[i].item()), int(truth_cen_y[i].item())), color=(0, 255, 0), radius=0, thickness=-1)

        cv.rectangle(img, [int(bb_uL_x[i].item()), int(bb_uL_y[i].item())], [int(bb_lR_x[i].item()), int(bb_lR_y[i].item())], color=(255, 255, 255))

        cv.line(img, [int(pred_cen_x[i].item()), int(pred_cen_y[i].item())], [int(truth_cen_x[i].item()), int(truth_cen_y[i].item())], color=(255, 255, 255) )
        cv.line(img, [int(bb_uL_x[i].item()), int(bb_uL_y[i].item())], [int(bb_lR_x[i].item()), int(bb_lR_y[i].item())], color=(255, 255, 255))

        cv.imwrite(f"test_{i}.png", img)

    #y = torch.reshape(torch.as_tensor(np.asarray(boxes[0:10])), (10,4))
    #pred = torch.reshape(torch.as_tensor(np.asarray(boxes[10:20])), (10,4)) #81
    #pred = torch.reshape(torch.as_tensor(np.zeros((10, 4))), (10,4)) #81
    #loss = loss_fn(y, pred)
    #loss_1 = loss_fn(y, y)
    #print('Loss', loss, loss_1)
    #print(y)
    #print(pred)'''