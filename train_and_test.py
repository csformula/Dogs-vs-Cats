# -*- coding: utf-8 -*-

import os
import copy
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt



class Epoch_History(object):
    ''' Tracking and ploting loss and acc for each epoch. '''
    
    def __init__(self):
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
    def plot_history(self):
        self.epochs = list(range(len(self.train_loss_history)))
            
        # plot loss history
        fig1 = plt.figure(figsize=(15,5))
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Training and Validation Loss')
        plt.plot(self.epochs, self.train_loss_history, label='Training')
        plt.plot(self.epochs, self.val_loss_history, label='Validation')
        plt.legend()
        
        # plot acc history
        fig2 = plt.figure(figsize=(15,5))
        plt.title('Acc History')
        plt.xlabel('Epoch')
        plt.ylabel('Training and Validation Acc')
        plt.plot(self.epochs, self.train_acc_history, label='Training')
        plt.plot(self.epochs, self.val_acc_history, label='Validation')
        plt.legend()
    

def train_model(model, dataset, criterion, optimizer, lr_pattern=None, device=torch.device('cpu'), 
                num_epochs=100, history_tracker=Epoch_History(), suffix=''):
    '''
    Train and validate datasets. 
    Args:
        model(callable): the model to be trained.
        dataset(dict): dict containing training and validation sets loaders.
        criterion(callable): criterion / loss_fn.
        optimizer(class Optimizer): optimizer.
        lr_pattern(wrapper, optim.lr_scheduler): lr updating rule.
        device(torch.device): the device 
        num_epochs(int): num of epochs.
        history_tracker(class Epoch_History): tracking loss and acc for each epoch.
        suffix(str, optional): suffix to be added into filename of saved weights.
    '''
    
    start = time.time()
    model.to(device)
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        if lr_pattern is not None:
            lr_pattern.step()
        
        print(f'Epoch {epoch+1}/{num_epochs} processing...')
        
        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_acc = 0
            num_imgs = 0
            for i, sample in enumerate(dataset[mode]):
                imgs = sample[0].to(device)
                labels = sample[1].to(device)
                
                with torch.set_grad_enabled(mode=='train'):
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if mode=='train':
                        loss.backward()
                        optimizer.step()
                        if (i+1) % 50 == 0:
                            print(f'{(i+1)*len(labels):<5d} images processed...')
                    
                running_loss += loss * len(labels)
                running_acc += torch.sum(preds==labels)
                num_imgs += len(labels)
            
            epoch_loss = running_loss / num_imgs
            epoch_acc = running_acc.double().item() / num_imgs
            
            if mode == 'train':
                history_tracker.train_loss_history.append(epoch_loss)
                history_tracker.train_acc_history.append(epoch_acc)
            else:
                history_tracker.val_loss_history.append(epoch_loss)
                history_tracker.val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model.state_dict())
                
            print(f'Epoch {epoch+1} {mode} finished: loss {epoch_loss:<.3f}, acc {epoch_acc*100:<.0f}%')
        
        print('-'*30)
    
    end = time.time()
    torch.save(best_model, './weights_'+suffix+'.pt')
    num_imgs_total = len(dataset['train'].dataset) + len(dataset['val'].dataset)
    
    print(f'Training finished, best state dict saved!')
    print(f'{"Images trained:":<17s} {num_imgs_total:<d}')
    print(f'{"Best acc:":<17s} {best_acc*100:<.0f}%')
    print(f'{"Num of epochs:":<17s} {num_epochs:<d}')
    print(f'{"Total time:":<17s} {end-start:<.3f}s')
    print(f'{"Time per epoch:":<17s} {(end-start)/num_epochs:<.3f}s')    
    print(f'{"Time per image:":<17s} {(end-start)/num_epochs/num_imgs_total:<.3f}s')
          
    return history_tracker



def test_model(model, dataset, fivecrop=False, device=torch.device('cpu'), suffix=''):
    ''' 
    Test data and saved the result into a csv file. 
    Args:
        model(callable): the model applied to test data.
        dataset(class Dataloader): Dataloader containing test set.        
        fivecrop(bool, optional): If true, fivecrop input data.
        device(torch.device, optional): the device
        suffix(str, optional): suffix to be added into filename of saved csv.
    '''
    
    start = time.time()
    model.to(device)
    model.eval()
    preds = []
        
    for idx, image in dataset:
        image = image.to(device)
        # testing data one by one
        if fivecrop==False:
            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)
        # testing data with fivecrop
        else:
            bs, n, c, h, w = image.size()
            image = image.view(bs*n, c, h, w)
            with torch.no_grad():
                output = model(image)
                output = output.view(bs, n, -1).mean(1)
                _, pred = torch.max(output, 1)
                
        preds.append((idx.item(), pred.item()))

        if len(preds)%1000 == 0:
            print(f'{len(preds)} images processed ...')

    results = sorted(preds, key=lambda x: x[0], reverse=False)
    results_df = pd.DataFrame(results, columns=('id', 'label'))
    if fivecrop==False:
        results_df.to_csv('mysub_nofive_'+suffix+'.csv', header=True, index=False, sep=',')
    else:
        results_df.to_csv('mysub_five_'+suffix+'.csv', header=True, index=False, sep=',')
    
    end = time.time()
    
    print('-'*30)
    print('Test finished, submission csv saved!')
    print(f'{"Images tested:":<17s} {len(preds):<d}')
    print(f'{"Total time:":<17s} {end-start:<.3f}s')
    print(f'{"Time per image:":<17s} {(end-start)/len(preds):<.3f}s')

    return results
