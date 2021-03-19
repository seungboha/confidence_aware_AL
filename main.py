from model import resnet
from model import densenet_BC
from model import vgg

import data_new as dataset
import crl_utils
import metrics
import utils
import train

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datetime

parser = argparse.ArgumentParser(description='Confidence Aware Learning')
parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--data', default='cifar10', type=str, help='Dataset name to use [cifar10, cifar100, svhn]')
parser.add_argument('--model', default='res', type=str, help='Models name to use [res, dense, vgg]')
parser.add_argument('--rank_target', default='softmax', type=str, help='Rank_target name to use [softmax, margin, entropy]')
parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--data_path', default='../data/', type=str, help='Dataset directory')
parser.add_argument('--save_path', default='./test/', type=str, help='Savefiles directory')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lr', default=0.1, type=float)

parser.add_argument('--initial_budget', default=2000, type=int, help='The num of initially labeled points')
parser.add_argument('--budget', default=2000, type=int, help='The num of acquisition every AL cycle')
parser.add_argument('--max_budget', default=20000, type=int, help='The num of total acquisition')
parser.add_argument('--subset', default=10000, type=int, help='The num of subset of unlabeled pool where acquisition is done')

args = parser.parse_args()

def get_confidence(model, dataloader):
    # choose data points with smallest confidence
    model.eval()
    unlabeled_confidence = torch.Tensor([]).cuda()

    with torch.no_grad():
        for images, _, _ in dataloader:
            images = images.cuda()

            output = model(images)
            prob = F.softmax(output, dim=-1)
            confidence, _ = prob.max(dim=-1)
            unlabeled_confidence = torch.cat((unlabeled_confidence, confidence), 0)

    return unlabeled_confidence.cpu()


    #_, topk_indices = torch.topk(unlabeled_confidence, k=budget, largest=False)
    #topk_indices = topk_indices.tolist()

    #return np.asarray(unlabeled_indices)[topk_indices]

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)

def main():
    # set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    # check save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # set num_class
    if args.data == 'cifar100':
        num_class = 100
    else:
        num_class = 10

    # set num_classes
    model_dict = {
        "num_classes": num_class,
    }

    _, test_loader, \
        test_onehot, test_label = dataset.get_loader(args.data,
                                        args.data_path,
                                        args.batch_size)
    
    train_set = dataset.get_dataset(args.data,
                                    args.data_path, 
                                    mode='train')
    unlabeled_pool = dataset.get_dataset(args.data,
                                    args.data_path, 
                                    mode='unlabeled')
    num_train = len(train_set)

    indices = list(range(num_train))
    random.shuffle(indices)

    labeled_set = indices[:args.initial_budget]
    unlabeled_set = indices[args.initial_budget:]

    labeled_dataloader = DataLoader(train_set, 
                                    sampler=SubsetRandomSampler(labeled_set),
                                    batch_size = args.batch_size,
                                    drop_last = True)


    now = datetime.datetime.now()
    formatedDate = now.strftime('%Y%m%d_%H_%M_')
    result_logger = utils.Logger(os.path.join(args.save_path, formatedDate +'result.log'))

    arguments = []
    for key, val in (args.__dict__.items()):
        arguments.append("{} : {}\n".format(key, val))
    result_logger.write(arguments)
    result_logger = utils.Logger(os.path.join(args.save_path, formatedDate +'result.log'))
    # make logger
    train_logger = utils.Logger(os.path.join(save_path, formatedDate + 'train.log'))
    test_epoch_logger = utils.Logger(os.path.join(save_path, formatedDate + 'test_epoch.log'))

    current_train = len(labeled_set)
    while(current_train < args.max_budget + 1):
        # set model
        if args.model == 'res':
            model = resnet.ResNet152(**model_dict).cuda()
        elif args.model == 'dense':
            model = densenet_BC.DenseNet3(depth=100,
                                        num_classes=num_class,
                                        growth_rate=12,
                                        reduction=0.5,
                                        bottleneck=True,
                                        dropRate=0.0).cuda()
        elif args.model == 'vgg':
            model = vgg.vgg16(**model_dict).cuda()

        # set criterion
        cls_criterion = nn.CrossEntropyLoss().cuda()
        ranking_criterion = nn.MarginRankingLoss(margin=0.0).cuda()

        # set optimizer (default:sgd)
        optimizer = optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=0.0005,
                            nesterov=False)

        # set scheduler
        scheduler = MultiStepLR(optimizer,
                                milestones=[120,160],
                                gamma=0.1)

        # make History Class
        correctness_history = crl_utils.History(len(labeled_dataloader.dataset))

        # start Train
        for epoch in range(1, args.epochs + 1):
            train.train(labeled_dataloader,
                        model,
                        cls_criterion,
                        ranking_criterion,
                        optimizer, 
                        epoch,
                        correctness_history,
                        train_logger,
                        args)
            test_acc, test_loss = metrics.evaluate(test_loader, model,cls_criterion, args.budget, epoch, test_epoch_logger)
            scheduler.step()
            # save model
            if epoch == args.epochs:
                torch.save(model.state_dict(),
                        os.path.join(save_path, 'model.pth'))
        # finish train

        # calc measure
        acc, aurc, eaurc, aupr, fpr, ece, nll, brier = metrics.calc_metrics(test_loader,
                                                                            test_label,
                                                                            test_onehot,
                                                                            model,
                                                                            cls_criterion)
        # result write
        result_logger.write([current_train, test_acc, aurc*1000, eaurc*1000, aupr*100, fpr*100, ece*100, nll*10, brier*100])
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:args.subset]
        unlabeled_poolloader = DataLoader(unlabeled_pool,
                                        sampler = SubsetSequentialSampler(subset),
                                        batch_size = args.batch_size,
                                        drop_last = False)
        all_confidence = get_confidence(model, unlabeled_poolloader)
        print(len(all_confidence))
        arg = np.argsort(all_confidence)
        labeled_set = list(set(labeled_set) | set(np.array(unlabeled_set)[arg][:args.budget]))
        unlabeled_set = list(set(unlabeled_set) - set(labeled_set))
        current_train = len(labeled_set)

        #unlabeled_set = list(torch.tensor(unlabeled_set)[arg][args.budget:].numpy()) \
        #                            + unlabeled_set[args.subset:]
        print("after acquistiion")
        print('current labeled :', len(labeled_set) )
        print('current unlabeled :', len(unlabeled_set) )

        labeled_dataloader = DataLoader(train_set,
                                        sampler = SubsetRandomSampler(labeled_set),
                                        batch_size = args.batch_size,
                                        drop_last = True)
if __name__ == "__main__":
    main()



