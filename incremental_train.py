import os
import sys
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from dynamic_model import param_per_task_group_helper


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

def train_epoch(model, data_loader, opti, t):
    epoch_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    model.train()
    nr_batches = 0
    for _, (x,y) in tqdm(enumerate(data_loader)):
        nr_batches += 1
        x, y = x.to(device), y.to(device)
        opti.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opti.step()
        epoch_loss+=loss.item()
    return epoch_loss/float(nr_batches)

def accuracy_epoch(model, data_loaders, t):
    model.eval()
    correct_cnt, total_cnt = 0, 0
    with torch.no_grad():
        for _, (x, y) in enumerate(data_loaders):
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1, keepdim=True)
            correct_cnt += pred.eq(y.view_as(pred)).sum().item()
            total_cnt += x.data.size()[0]
    return float(correct_cnt)/float(total_cnt)

def train(c, m, d):
    data_path = '../data'
    #model_path = '../models'

    learning_rate = c['learning_rate']
    nr_epochs = c['nr_epochs']
    batch_size = c['batch_size']
    nr_tasks = c['nr_tasks']
    layer_list_info = c['layer_list_info']

    data_obj = d(data_path, nr_tasks)
    model = m(layer_list_info)

    for task_id in range(nr_tasks):
        print("Training for task {}".format(task_id))
        train_loader, test_loader = data_obj.get_data(task_id, batch_size)
        model.add_model_for_task(task_id)
        optimizer = optim.Adam(param_per_task_group_helper(task_id, model), lr=learning_rate)
        for epoch in tqdm(range(1, nr_epochs)):
            train_loss = train_epoch(model, train_loader, optimizer, task_id)
            test_accuracy = accuracy_epoch(model, test_loader, task_id)
            print(task_id, epoch, train_loss, test_accuracy)

def test():
    config = {'learning_rate': 0.01,
              'nr_epochs': 5,
              'batch_size': 128,
              'layer_list_info': {
                    0: [('Conv2d', 3, 16, 3), ('BatchNorm2d', 16), ('ReLU',), ('MaxPool2d', 2, 2)],
                    1: [('Conv2d', 16, 4, 3), ('BatchNorm2d', 4), ('ReLU',), ('MaxPool2d', 2, 2)]
                },
               'nr_tasks': 5
    }
    from dynamic_model import DynaNet as model_def
    from dataset import dataset
    train(config, model_def, dataset)

if __name__ == "__main__":
    test()
