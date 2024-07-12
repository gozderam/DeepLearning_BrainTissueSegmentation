import torch
import torch.optim as optim
import pandas as pd
import json


def train_network(train_loader, val_loader, network, optimizer_name, 
    learning_rate, loss_fn, batch_size, 
    epochs, load_model, model_path, loss_path, device):

    # some large value for initial best validation loss
    best_vloss = 1_000_000.

    # train loss values after each epoch and batch
    epoch_tloss_values = [] 
    batch_tloss_values = []

    # validation loss values after each epoch
    epoch_vloss_values = []

    # initally set to zero, if load_model, value will be changed
    epoch = 0 

    # define the model, optimizer, loss
    net = network.to(device).train()
    if optimizer_name == "adam":    
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # loading previously stored model if load_model
    if load_model:
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']+1
        epoch_tloss_values = checkpoint['epoch_tloss_values']
        batch_tloss_values = checkpoint['batch_tloss_values']
        epoch_vloss_values = checkpoint['epoch_vloss_values']
        print('Model loaded from checkpoint')

    # train model
    for e in range(epoch, epochs):
        print(f'  Epoch: {e}')
        epoch_tloss_value = __train_one_epoch(train_loader, net, optimizer, loss_fn, e, device, batch_tloss_values) 
        epoch_tloss_values.append(epoch_tloss_value)
        print(f'Training in epoch {e} finished. Train loss: {epoch_tloss_value}]')

        with torch.no_grad():
            running_vloss = 0.0
            for i, batch in enumerate(val_loader):
                vX, vy = batch
                vX, vy = vX.to(device), vy.to(device)
                vpreds = network(vX)
                vloss = loss_fn(vpreds, vy).item()
                running_vloss += vloss
                del vX, vy, vpreds

        avg_vloss = running_vloss / (i+1)
        epoch_vloss_values.append(avg_vloss)
        print(f'[Validation in epoch {e} finished. Validation loss: {avg_vloss:.3f}]')

        if avg_vloss < best_vloss:

            torch.save({
                'model_state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': e,
                'epoch_tloss_values': epoch_tloss_values,
                'batch_tloss_values': batch_tloss_values,
                'epoch_vloss_values': epoch_vloss_values
            }, model_path)

    
    # save loss
    __save_loss(epoch_tloss_values, epoch_vloss_values, batch_tloss_values, loss_path) 

def save_params(model_name, axis, optimizer_name, loss_fn, batch_size, epochs, img_height, img_width, parms_path):
    loc = locals()
    loc.pop('parms_path')
    with open(f"{parms_path}.json", "w") as outfile:
        json.dump(loc, outfile)


def __save_loss(epoch_tloss, epoch_vloss, batch_tloss, loss_path):
    epoch_loss_df = pd.DataFrame({'epoch': list(range(len(epoch_tloss))), 'train_loss': epoch_tloss, 'val_loss': epoch_vloss})
    batch_tloss_df = pd.DataFrame({'batch': list(range(len(batch_tloss))), 'batch_train_loss': batch_tloss})
    epoch_loss_df.to_csv(loss_path+"epoch_train_val.csv", index=False)
    batch_tloss_df.to_csv(loss_path+"batch_train.csv", index=False)

def __train_one_epoch(data_loader, network, optimizer, loss_fn, epoch, device, batch_tloss_values, print_by_batches = 16):

    running_loss = 0.0
    last_loss = 0.0

    for i, batch in enumerate(data_loader):
        X, y = batch
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        preds = network(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_tloss_values.append(loss.item())

        del X, y, preds

        if i % print_by_batches == (print_by_batches - 1):
            last_loss = running_loss / print_by_batches
            print(f'[epoch: {epoch}, batches: {i + 1 - print_by_batches:5d} - {i:5d}] train loss: {last_loss:.3f}')
            running_loss = 0.0

    return last_loss

