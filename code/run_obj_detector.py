from objectDataset import BasicDataset
from objectDetector import BasicNetwork, train_loop, test_loop
#from conv_network import BasicNetwork, train_loop, test_loop
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import torch
import wandb

from CIoU import CIoU


if __name__ == "__main__":
    wandb.init(project="CS545_final", entity="jcscheufele")
    new_name = "New Data Test :)"
    wandb.run.name = new_name
    wandb.run.save()

    '''tr_dataset = BasicDataset(0, 145)
    torch.save(tr_dataset, "tr_dataset.pt")
    print("tr data saved")

    val_dataset = BasicDataset(145, 175)
    torch.save(val_dataset, "val_dataset.pt")
    print("val data saved")

    te_dataset = BasicDataset(175, 206)
    torch.save(te_dataset, "te_dataset.pt")
    print("te data saved")'''
    

    
    tr_dataset = torch.load("tr_dataset.pt")
    print("tr data loaded")

    val_dataset = torch.load("val_dataset.pt")
    print("val data loaded")
    


    shuffle = True
    batch_size = 1
    epochs = 10000
    learningrate = 1e-5 #1e-3

    wandb.config = {
    "learning_rate": learningrate,
    "epochs": epochs,
    "batch_size": batch_size
    }

    train_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print(len(train_loader.dataset))
    print(len(valid_loader.dataset))

    out_features = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = BasicNetwork(out_features).to(device)
    print(model)

    #loss_fn = nn.MSELoss(reduction='mean')
    loss_fn = CIoU()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    tr_key = 0
    te_key = 0
    will_save = False
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        if (epoch % int(10)) == 0:
            will_save = True
        else:
            will_save = False

        training_dicts, train_loss, train_error, tr_key = train_loop(train_loader, model, loss_fn, optimizer, device, epoch, batch_size, will_save, tr_key)
        testing_dicts, test_loss, test_error, te_key = test_loop(valid_loader, model, loss_fn, device, epoch, batch_size, will_save, te_key)
        if will_save:
            for dict1, dict2 in zip(training_dicts, testing_dicts):
                wandb.log(dict1)
                wandb.log(dict2)
        wandb.log({"Epoch Training Loss":train_loss, "Epoch Testing Loss": test_loss,
        "Epoch Training Percent Error": train_error, "Epoch Testing Percent Error": test_error, 
        "Epoch epoch":epoch})

    save_loc = f"../../data/models/new/model_{new_name}.pt"
    print(f"saving Network to {save_loc}")
    torch.save(model, save_loc)