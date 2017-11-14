"""
Main code for weakly supervised object localization
===================================================
*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import torch
import os
import numpy as np
import time
import datetime
from model.model import WSL, load_pretrained
import data_utils.load_voc as load_voc
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from spn_codes.models import SPNetWSL
from evaluate.rst_for_corloc import rst_for_corloc
from evaluate.corloc_eval import corloc


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=64, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-log', default=False,
                    help='disable logging while training')
parser.add_argument('--gpuID', default=0, type=int,
                    help='GPU ID')
parser.add_argument('--ck-pt', default='/disk3/zhangyu/WeaklyDetection/spn_new/\
checkpt/best_model/best_checkpoint_epoch20.pth.tar',
                    help='directory of check point will be used in test time')

data_dir = '/home/zhangyu/data/VOC2007/'
# voc_test = '/home/zhangyu/data/VOC2007_test/'
root_dir = '/disk3/zhangyu/WeaklyLoc/spn_train_by_me_bicubic_intep/'
imgDir = os.path.join(data_dir, 'JPEGImages')
train_annos = os.path.join(data_dir, 'train_annos')
trainval_annos = os.path.join(data_dir, 'Annotations')
att_map_dir = os.path.join(root_dir, 'results/atten_map_trainval/')
cls_number = 20

save_file = os.path.join(att_map_dir, 'predict{}.csv'.format(
    datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))


def main():
    global args
    global gpuID
    args = parser.parse_args()
    gpuID = args.gpuID
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.cuda = 0
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    num_class = 20
    net = WSL(num_class)
    load_pretrained(model=net, fname=args.ck_pt)
    if args.cuda:
        net.cuda(gpuID)
    test_loader = prepare_data(trainval_annos)

    # global ft
    # ft = torch.zeros(args.batch_size, 1024, 14, 14)
    test(test_loader, net)
    corloc_rst = corloc(save_file, trainval_annos)
    print('Corloc results: {}'.format(corloc_rst))


def test(test_loader, model):
    batch_time = AverageMeter()
    accu = AverageMeter()

    # switch to evaluation mode
    model.eval()

    end = time.time()
    for i, data in enumerate(test_loader):
        print('Testing: [{0}/{1}] '.format(i, len(test_loader)))
        batch_names = data['filename']
        img_szs = data['sz']
        input = data['image'].float()
        target = data['class'].float()
        if args.cuda:
            target = target.cuda(gpuID)
            input_var = torch.autograd.Variable(input, volatile=True).cuda(gpuID)
        else:
            input_var = torch.autograd.Variable(input, volatile=True)

        # compute output: cls_scores, ft as last conv feature, and proposals 
        cls_scores, ft = model.get_att_map(input_var)
        lr_weigth = model.classifier[1].weight.cpu().data.numpy()
        # convert to npy 
        img_szs = img_szs.numpy()
        scores = cls_scores.cpu().data.numpy()
        # props = proposals.cpu().numpy()
        feature = ft.cpu().data.numpy()
        targets = target.cpu().numpy()
        # generate results
        rst_for_corloc(batch_names, targets, img_szs, scores, feature,
                       lr_weigth, att_map_dir, save_file)

        # measure accuracy and record loss
        acc = accuracy(cls_scores, target)
        accu.update(acc)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            info = 'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Accuracy {accu.val:.4f} (avg:{accu.avg:.4f}) '.format(accu=accu)
            print(info)
    return accu.avg


# def get_sp_forward(self, input, output):
#     ft.copy_(output.data)


def prepare_data(annos_path):
    # prepare dataloader for training and validation
    train_dataset = load_voc.VOCDataset(
        xmlsPath=annos_path, imgDir=imgDir,
        transform=transforms.Compose([
            load_voc.Rescale((224, 224)),
            load_voc.ToTensor(),
            load_voc.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, drop_last=True)
    return train_loader


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
