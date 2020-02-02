import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt

from model import UNet
from dataloader import DataLoader
import torch.nn.functional as F

def train_net(net,
              epochs=5,
              data_dir='data/cells/',
              n_classes=2,
              lr=0.01,
              val_percent=0.1,
              save_cp=True,
              gpu=True):
    loader = DataLoader(data_dir)

    N_train = 2*loader.n_train()
 
    optimizer = optim.SGD(net.parameters(),
                            lr=lr,
                            momentum=0.99,
                            weight_decay=0.0005)

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(net.parameters(), lr=0.01)

        for i, (img, label) in enumerate(loader):
            shape = img.shape
            #print(shape)
            
            # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width) 1,1,h,w
            image = torch.tensor(img)
            image = image.resize_((1, 1, image.size(0), image.size(1)))
            
            label = torch.tensor(label)

            #print(image.size())
            # todo: load image tensor to gpu
            if gpu:
                image = image.cuda()
                label = label.cuda()
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #print(device)
            #image.to(device = device, dtype=torch.float32)

            # todo: get prediction and getLoss()
            
            pred_label = net(image.float()) 
            
            ## crop label
            diff = label.size(0) - pred_label.size(2)
            
            if diff % 2 == 0:
                diff = int(diff/2)
                label = label[diff:(label.size(0)-diff), diff:(label.size(1)-diff)]
            else:
                diff = int((diff)/2)
                label = label[diff:(label.size(0)-diff-1), diff:(label.size(1)-diff-1)]
        
            
            label = label.reshape((1, label.size(0), label.size(1))).long()
            
            #print(pred_label.size()) # torch.Size([1, 2, 256, 256])       
            #print(label.size()) #torch.Size([1, 256, 256])
            
            loss = getLoss(pred_label, label)
            #loss = criterion(pred_label, label)
            
            optimizer.zero_grad()            
            loss.backward()

            epoch_loss += loss.item()
            
            optimizer.step()
 
            print('Training sample %d / %d - Loss: %.6f' % (i+1, N_train, loss.item()))

            # optimize weights

        torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))

    # displays test images with original and predicted masks after training
    loader.setMode('test')
    net.eval()
    with torch.no_grad():
        for _, (img, label) in enumerate(loader):
            shape = img.shape
            img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
            if gpu:
                img_torch = img_torch.cuda()
                
            #img_torch = img_torch.resize_((1, 1, 256, 256))
            #label = torch.tensor(label)
            #label = label.resize(256,256)
            
            pred = net(img_torch)
            #print(pred.size()) ## [1,2,256,256]

            pred_sm = softmax(pred)  # [1, 2, 256, 256]    
            #pred_sm = F.softmax(pred,dim=1)
            
            _,pred_label = torch.max(pred_sm,1) #[1,256,256]

            plt.subplot(1, 3, 1)
            plt.imshow(img*255.)
            plt.subplot(1, 3, 2)
            plt.imshow(label*255.)
            plt.subplot(1, 3, 3)
            plt.imshow(pred_label.cpu().detach().numpy().squeeze()*255.)
            plt.show()
            #print("END")
        print("OUT OF LOOP")

def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    #print(p.size())
    return cross_entropy(p, target_label)

def softmax(input):  # input [1,2,256,256]
    # todo: implement softmax function 
    input = input.reshape((input.size(1), input.size(2), input.size(3)))
    exp = torch.exp(input) #[2,256,256]
    sum_exp = torch.sum(exp,0) #[256,256]
    sum_exp = sum_exp.reshape((1, 1, sum_exp.size(0), sum_exp.size(1)))
    #p = exp/sum_exp
    
    return exp/sum_exp  # output [1,2, 256, 256]

def cross_entropy(input, targets): # [1,2,256,256] [1,256,256]
    # todo: implement cross entropy
    # Hint: use the choose function
    
    ## input [1,2,h,w]
    ## targets [1,h,w]
    #targets = torch.tensor(targets)
    M = input.size(2) * input.size(3)
    pred = choose(input, targets)
    ce = torch.sum(-1*torch.log(pred))/M
    
    #targets = targets.reshape((1, targets.size(0), targets.size(1), targets.size(2)))
    
    #M = input.size(1) * input.size(2)
    #ce = -1 * torch.sum(targets * torch.log(input))
    # = choose(input, targets)

    return ce

# Workaround to use numpy.choose() with PyTorch
def choose(pred_label, true_labels): #[1,2,256,256] [1,256,256]
    size = pred_label.size() #[1,2,256,256]
    ind = np.empty([size[2]*size[3],3], dtype=int)
    i = 0
    for x in range(size[2]):
        for y in range(size[3]):
            ind[i,:] = [true_labels[0,x,y], x, y]
            i += 1
    
    pred = pred_label[0,ind[:,0],ind[:,1],ind[:,2]].view(size[2],size[3])

    return pred
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/cells/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        n_classes=args.n_classes,
        gpu=args.gpu,
        data_dir=args.data_dir)
