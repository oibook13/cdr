'''
    Training code for Tiny ImageNet
'''
import sys,os
import numpy as np
import argparse
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from src.cdr import CDR
from src.utils import *

# extract intermediate features for initialization
intm_feats = None
def init_hook(module, input, output):
    global intm_feats
    intm_feats = input[0].data.detach()

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')

parser.add_argument('--dataset', default='path to dataset', type=str)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--num_class', type=int, default=200, help='num of classes')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=64, type=int, metavar='N',
                    help='batch size for training')
parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                    help='batch size for testing')
parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_scheduler', type=int, nargs='+', default=[10, 20],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_decay', type=float, default=0.1, help='LR is decreased by lr_decay on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--wc', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint')

parser.add_argument('--model', type=str, default='resnet101',
                    help='model used to train, options: vgg16 resnet50 resnet101 densenet169')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='initialize with pre-trained model')

parser.add_argument('--drm_type', default='fc', type=str, help='dimensionality reduction module: fc | cdr')
parser.add_argument('--fc_kaiming_alpha', default=0.0, type=float, help='alpha in kaiming initialization using uniform distribution')
parser.add_argument('--cdr_alpha', default=0.01, type=float, help='influence control factor for distance to the other centroids CDR')
parser.add_argument('--cdr_normalized_radius', default=1.0, type=float, help='initial bound in CDR')
parser.add_argument('--cdr_t', default=1.0, type=args_str2list, help='temperature in CDR')
parser.add_argument('--cdr_p', default=None, type=args_str2list, help='Lp distance metrics in CDR')
parser.add_argument('--cdr_num_init_samples', default=-1, type=int, help='a integer indicates initialize centroids by the mean of featuares, -1 indicate using random numbers')
parser.add_argument('--input_img_size', default=224, type=int, help='input image size to a model')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Set random seed
seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

modelzoo = {
    'densenet169': models.densenet169,
    'vgg16': models.vgg16,
    'resnet101': models.resnet101,
    'resnet50': models.resnet50,
    'resnet34': models.resnet34,
    'resnet18': models.resnet18,
}

best_acc = 0  # best test accuracy
def main():
    global best_acc

    if not os.path.isdir(args.checkpoint):
        #os.mkdir(args.checkpoint)
        os.makedirs(args.checkpoint, exist_ok=True)

    # Data loading code
    traindir = os.path.join(args.dataset, 'train')
    valdir = os.path.join(args.dataset, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(args.input_img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    valdir = os.path.join(args.dataset, 'val', 'images')
    valgtfile = os.path.join(args.dataset, 'val', 'val_annotations.txt')
    val_dataset = TImgNetDataset(valdir, valgtfile, class_to_idx=train_loader.dataset.class_to_idx.copy(),
            transform=transforms.Compose([
            transforms.Resize(args.input_img_size),
            transforms.ToTensor(),
            normalize,
            ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    model = modelzoo[args.model](pretrained=args.pretrained)
    modeltype = type(model).__name__
    # update dimensionality reduction module
    model = updateDimensionalityReductionModule(model, args.num_class, args)

    # inittialize centroids from datapoints
    if args.drm_type == 'cdr' and args.cdr_num_init_samples > 0:
        print('initialize dfc with sampled {} features'.format(args.cdr_num_init_samples))
        num_init_samples = args.cdr_num_init_samples
        feat_wrt_label = {}
        init_hook_handle = model.fc.register_forward_hook(init_hook)
        model = model.cuda()
        with torch.no_grad():
            for samples, target in train_loader:
                samples, target = samples.cuda(), target.cuda()
                target = target.cpu().numpy()
                output = model(samples)
                for i in range(target.shape[0]):
                    if feat_wrt_label.get(target[i]) is None:
                        feat_wrt_label.update({target[i]:[]})
                    if len(feat_wrt_label[target[i]]) < num_init_samples:
                        feat_wrt_label[target[i]].append(intm_feats[i,:])
        init_datapoints = []
        for key, value in feat_wrt_label.items():
            init_datapoints = torch.stack(value,dim=0).mean(dim=0)
            init_datapoints /= torch.norm(init_datapoints, p=2)
            init_datapoints *= args.cdr_initbound
            model.fc.metrics[key].basis = torch.nn.Parameter(init_datapoints.unsqueeze(0))
        if init_hook_handle is not None:
            init_hook_handle.remove()

    model = torch.nn.DataParallel(model).cuda()

    print('Model {} params num: {:.2f}M'.format(args.model, sum(p.numel() for p in model.parameters())/1e+6))
    print(args)

    # initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wc)

    # training and testing
    for epoch in range(args.epochs):
        adjust_lr(optimizer, epoch, args.lr_decay, args.lr_scheduler)

        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        test_loss, test_acc = test(val_loader, model, criterion)
        print('Epoch [{} | {}]: train_loss: {:.4f}, train_err: {:.4f}, test_loss: {:.4f}, test_err: {:.4f}'.format(
            epoch + 1, args.epochs,
            train_loss, 1-train_acc,
            test_loss, 1-test_acc))

        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'err': 1-test_acc,
                'acc': test_acc,
                'best_acc': best_acc,
                'best_err': 1-best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    print('Best err: {:.4f}'.format(1-best_acc))

def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.cuda(), targets.cuda(async=True)
        # inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        preds = torch.softmax(outputs.detach(),dim=1)

        # evaluate accuracy and loss
        cur_acc = accuracy(outputs.data, targets.data)[0]
        losses.update(loss.item(), inputs.size(0))
        acc.update(cur_acc.item(), inputs.size(0))

        # compute gradient and update the parameters of the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, acc.avg)

def test(val_loader, model, criterion):
    global best_acc

    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()
    for batch_idx, (inputs, targets) in enumerate(val_loader):

        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        preds = torch.softmax(outputs.detach(),dim=1)

        # evaluate accuracy and loss
        cur_acc = accuracy(outputs.data, targets.data)[0]
        losses.update(loss.item(), inputs.size(0))
        acc.update(cur_acc.item(), inputs.size(0))

    return (losses.avg, acc.avg)

def updateDimensionalityReductionModule(model, num_class, opts):
    """
        Modify the last fully-connected layer with a CDR module or a new fully-connected layer (to adapt to new dataset)
        Input
            model: the model used to train
            num_class: number of classes
            opts: hyperparameters
        Output
            a modified model
    """
    tempList = list(model.children())
    module_name = tempList[-1].__class__.__name__
    assert (module_name == 'Linear'), '# error, the last layer should be fc/linear'
    if type(model).__name__=='ResNet':
        in_features = model.fc.in_features
  
        # if the input size to the model is 64, it needs ajust pooling layer accordingly
        # if the input size is 224, then no need to change for pooling layer
        if opts.input_img_size == 64:
            model.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

        if opts.drm_type == 'fc':
            t_layer = nn.Linear(in_features=in_features, out_features=num_class)
            # if opts.fc_kaiming_alpha > 0:
            #     nn.init.kaiming_uniform_(t_layer.weight.data, a=opts.fc_kaiming_alpha)
        elif opts.drm_type == 'cdr':
            t_layer = CDR(in_features=in_features, 
                            out_features=num_class, 
                            alpha=opts.cdr_alpha, 
                            normalized_radius=opts.cdr_normalized_radius, 
                            p=opts.cdr_p, t=opts.cdr_t)

        model.fc = t_layer

    return model

if __name__ == '__main__':
    main()
