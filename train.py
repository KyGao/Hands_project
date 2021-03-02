from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import argparse
import time

from models import *
from utils  import *
from dataloader import get_data_loader


class ClassifyModel(nn.Module):
    def __init__(self, num_class, model):
        super(ClassifyModel, self).__init__()

        models = {'resnet9': ResNet9(),
                'resnet18' : ResNet18(),
                'resnet34' : ResNet34(),
                'resnet50' : ResNet50(),
                'resnet101': ResNet101(),
                'resnet152': ResNet152(),
                'handsresnet':HandsResNet()}
        model = models[model]
        
        self.resnet_rgb = model
        self.resnet_depth = model
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_class)

    def forward(self, rgb, depth):
        rgb = self.resnet_rgb(rgb)
        depth = self.resnet_depth(depth)
        x = torch.cat((rgb, depth), 1)
        x = self.globalavgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu, batch_size, output_path):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        begin_time = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        fp.write("\n Epoch: " + str(epoch+1) + "/" + str(num_epochs) + "\n----------\n")

        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            count_batch = 0
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloders[phase]:
                count_batch += 1
                # get the inputs
                rgb, depth, labels = data

                # wrap them in Variable
                if use_gpu:
                    rgb = rgb.cuda()
                    depth = depth.cuda()
                    labels = labels.cuda()

                # backward + optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(rgb, depth)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    

                if phase == 'val':
                    with torch.no_grad():
                        optimizer.zero_grad()
                        outputs = model(rgb, depth)
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)


                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

                # print result every 10 batch
                if count_batch%10 == 0:
                    batch_loss = running_loss / (batch_size*count_batch)
                    batch_acc = running_corrects / (batch_size*count_batch)
                    fp.write(str(phase) + " Epoch [" + str(epoch+1) + "] Batch [" + str(count_batch) + "] Loss: " + str(batch_loss) + 
                            " Acc: " + str(batch_acc.item()) + " Time: " + str(time.time()-begin_time) + "s \n")
                    print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                          format(phase, epoch+1, count_batch, batch_loss, batch_acc, time.time()-begin_time))
                    begin_time = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            fp.write(str(phase) + " Loss: " + str(epoch_loss) + " Acc: " + str(epoch_acc.item()) + "\n")
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save model
            if phase == 'train':
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                torch.save(model,output_path+ '/resnet_on_PV_epoch{}.pkl'.format(epoch+1))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    fp.write("Training complete in " + str(time_elapsed // 60) + "m " + str(time_elapsed) + "s\n")
    fp.write("Best val Acc: " + str(best_acc) + "\n")


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch HANDS Training')
    parser.add_argument('--model',      default='resnet9', help='resnet9/18/34/50, wrn_40_2/_16_2/_40_1')
    parser.add_argument('--data_loc',   default='./labeled_data', type=str)
    parser.add_argument('--output_path',default='dumped', type=str)
    parser.add_argument('--log_file_name',   default='llllog.txt', type=str)
    # parser.add_argument('--checkpoint', default='resnet18', type=str)

    ### training specific args
    parser.add_argument('--batch_size',     default=32, type=int)
    parser.add_argument('--epoch_size',     default=10, type=int)
    parser.add_argument('--lr',         default=0.0001)

    args = parser.parse_args()



    
    print("Program start","-"*20)

    use_gpu = torch.cuda.is_available()

    batch_size = args.batch_size
    num_class = 31
    num_epoch = args.epoch_size

    #优化器参数
    learning_rate = args.lr
    #momentum = 0.9
    #学习率调整参数
    niter = int(num_epoch/2)
    niter_decay = num_epoch - niter


    fp = open(args.log_file_name, "w")

    fp.write("batch_size: " + str(batch_size) + "\nnum_class: " + str(num_class) + "\nnum_epoch: " + str(num_epoch) + "\n")
    fp.write("learning_rate: " + str(learning_rate) + "\nn_iter" + str(niter) + "\nn_iter_decay: " + str(niter_decay) + "\n-----------")

    print("batch_size:", batch_size, "num_classes:", num_class, "num_epoch:", num_epoch,
          "learning_rate:", learning_rate, "niter:", niter, "niter_decay:", niter_decay)


    dataloders = get_data_loader(args.data_loc, batch_size)

    # get model and replace the original fc layer with your fc layer
    print("get resnet model and init")
    model_ft = ClassifyModel(num_class, args.model)
    # if use gpu
    print("Use gpu:", use_gpu)
    if use_gpu:
        model_ft = model_ft.cuda()
    # model_ft = init_net(model_ft, init_type='xavier')



    print("Define loss function and optimizer......")
    # define cost function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # 学习率在epoch=niter后线性下降到0
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    exp_lr_scheduler = get_scheduler(optimizer_ft, niter, niter_decay)

    # train model
    print("start train_model......")
    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=num_epoch,
                           use_gpu=use_gpu,
                           batch_size=batch_size,
                           output_path = args.output_path)

    # save best model
    print("save model......")
    torch.save(model_ft, args.output_path+"/resnet_on_PV_best_total_val.pkl")

    fp.close()
