"""
Main code for weakly supervised object localization
===================================================
*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import torch
import torch.nn.functional as F
import os
import numpy as np
import shutil
import time
import datetime
from model.model import *
from spn_codes.models import SPNetWSL
import data_utils.load_voc as load_voc
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=64, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-log', default=False,
                    help='disable logging while training')
parser.add_argument('--gpuID', default=0, type=int,
                    help='GPU ID')

root_dir = '/home/zhangyu/data/VOC2007/'
save_root = '/disk3/zhangyu/WeaklyDetection/spn_new/'
imgDir = os.path.join(root_dir, 'JPEGImages')
train_annos = os.path.join(root_dir, 'train_annos')
val_annos = os.path.join(root_dir, 'val_annos')
vggParas = '/home/zhangyu/data/VGG_imagenet.npy'
# train_dir = '/home/zhangyu/data/tmp/'
check_point_dir = os.path.join(save_root, 'checkpt')
logging_dir = os.path.join(save_root, 'log')
if not os.path.isdir(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
if not os.path.isdir(check_point_dir):
    os.mkdir(check_point_dir)
if not os.path.isdir(os.path.join(check_point_dir, 'best_model')):
    os.mkdir(os.path.join(check_point_dir, 'best_model'))


def main():
    global args
    global log_file
    global gpuID
    log_file = os.path.join(logging_dir, 'log_{}.txt'.format(
        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    log_file_npy = os.path.join(logging_dir, 'log_{}.npy'.format(
        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    args = parser.parse_args()
    gpuID = args.gpuID
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.cuda = 0
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    num_class = 20
    net = SPNetWSL(num_class)

    if args.cuda:
        net.cuda(gpuID)

    train_loader, val_loader = prepare_data()

    # net = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    best_acc = 0
    train_loss = []
    train_loss_detail = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_epoch = 1
    for epoch in range(args.start_epoch, args.epochs):
        tr_avg_acc, tr_avg_loss, tr_detail_loss = \
            train(train_loader, net, optimizer, epoch)
        val_avg_acc, val_avg_loss = validation(val_loader, net)

        # save train/val loss/accuracy, save every epoch in case of early stop
        train_loss.append(tr_avg_loss)
        train_acc.append(tr_avg_acc)
        train_loss_detail += tr_detail_loss
        val_loss.append(val_avg_loss)
        val_acc.append(val_avg_acc)
        np.save(log_file_npy, {'train_loss': train_loss,
                               'train_accuracy': train_acc,
                               'train_loss_detail': train_loss_detail,
                               'val_loss': val_loss,
                               'val_accuracy': val_acc})

        # Save checkpoint
        is_best = val_avg_acc > best_acc
        best_acc = max(val_avg_acc, best_acc)
        save_file = os.path.join(
            check_point_dir, 'checkpoint_epoch{}.pth.tar'.format(epoch+1))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)
        if(is_best):
            tmp_file_name = os.path.join(check_point_dir, 'best_model',
                'best_checkpoint_epoch{}.pth.tar'.format(best_epoch))
            if os.path.isfile(tmp_file_name):
                os.remove(tmp_file_name)
            best_epoch = epoch + 1
            shutil.copyfile(save_file, os.path.join(
                check_point_dir, 'best_model',
                'best_checkpoint_epoch{}.pth.tar'.format(best_epoch)))


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accu = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    epoch_loss = []
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # prepare input
        input = data['image'].float()
        target = data['class'].float()
        if args.cuda:
            # input_var = torch.autograd.Variable(input)
            input_var = torch.autograd.Variable(input).cuda(gpuID)
            target_var = torch.autograd.Variable(target).cuda(gpuID)
            target = target.cuda(gpuID)
        else:
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        # hidden_maps.register_hook(lambda grad: print(grad.size()))
        output = model(input_var)
        # make_dot(output)
        # output = output.squeeze()
        loss = F.multilabel_soft_margin_loss(output, target_var)
        if args.cuda:
            loss = loss.cuda(gpuID)

        # measure accuracy and record loss
        acc = accuracy(output, target)
        accu.update(acc)
        losses.update(loss.data[0], input.size(0))
        epoch_loss.append(loss.data[0])
        batch_time.update(time.time() - end)
        end = time.time()

        # display and logging
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}][{1}/{2}] '.format(epoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Data {data_time.val:.3f} (avg:{data_time.avg:.3f}) '.format(data_time=data_time) + \
                   'Loss {loss.val:.4f} (avg:{loss.avg:.4f}) '.format(loss=losses) + \
                   'Accuracy {accu.val:.4f} (avg:{accu.avg:.4f})'.format(accu=accu)
            print(info)
            if not args.no_log:
                with open(log_file, 'a+') as f:
                    f.write(info + '\n')

        # output.register_hook(lambda grad: print(grad))
        # loss.register_hook(lambda  loss: print(loss))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # loss.backward(retain_graph=True)
        optimizer.step()

    # return loss, accuracy for recording and plotting
    return accu.avg, losses.avg, epoch_loss


def validation(val_loader, model):
    batch_time = AverageMeter()
    accu = AverageMeter()
    losses = AverageMeter()
    # switch to evaluation mode
    model.eval()

    end = time.time()
    for i, data in enumerate(val_loader):
        input = data['image'].float()
        target = data['class'].float()
        if args.cuda:
            input_var = torch.autograd.Variable(input, volatile=True).cuda(gpuID)
            target_var = torch.autograd.Variable(target, volatile=True).cuda(gpuID)
            target = target.cuda(gpuID)
        else:
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = F.multilabel_soft_margin_loss(output, target_var)
        if args.cuda:
            loss = loss.cuda(gpuID)

        # measure accuracy and record loss
        acc = accuracy(output, target)
        accu.update(acc)
        losses.update(loss.data[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            info = 'Test: [{0}/{1}] '.format(i, len(val_loader)) + \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:.4f} ({loss.avg:.4f}) '.format(loss=losses) + \
                   'Accuracy {accu.val:.4f} (avg:{accu.avg:.4f}) '.format(accu=accu)
            print(info)
            if not args.no_log:
                with open(log_file, 'a+') as f:
                    f.write(info + '\n')

    return accu.avg, losses.avg


def prepare_data():
    # prepare dataloader for training and validation
    train_dataset = load_voc.VOCDataset(
        xmlsPath=train_annos, imgDir=imgDir,
        transform=transforms.Compose([
            load_voc.Augmentation(),
            load_voc.Rescale((224, 224)),
            load_voc.ToTensor(),
            load_voc.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True)
    val_dataset = load_voc.VOCDataset(
        xmlsPath=val_annos, imgDir=imgDir,
        transform=transforms.Compose([
            # load_voc.Augmentation(),
            load_voc.Rescale((224, 224)),
            load_voc.ToTensor(),
            load_voc.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]))
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, drop_last=True)
    return train_loader, val_loader


def gen_loss_weight(target):
    """generate weight for loss, maybe not necessary"""
    positive_num = torch.sum(target, 1)
    class_num = torch.FloatTensor([target.size(1)]).cuda(gpuID) if args.cuda else torch.Tensor([target.size(1)])
    negative_num = class_num - positive_num
    weight = torch.div(negative_num, positive_num)
    weight = weight.expand((target.size(0), target.size(1)))
    return torch.mul(weight, target)


def load_pretrained(model, optimizer, fname):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))


def accuracy(output, target, threshold=0.5):
    """
    Compute precision for multi-label classification part
    accuracy = predict joint target / predict union target
    Use sigmoid function and a threshold to determine the label of output
    :param output: class scores from last fc layer of the model
    :param target: binary list of classes
    :param threshold: threshold for determining class
    :return: accuracy
    """
    sigmoid = torch.sigmoid(output)
    predict = sigmoid > threshold
    target = target > 0
    joint = torch.sum(torch.mul(predict.data, target))
    union = torch.sum(torch.add(predict.data, target) > 0)
    return joint / union


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
