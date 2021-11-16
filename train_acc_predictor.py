from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import horovod.torch as hvd
from tqdm import tqdm
import random
import time
import torch.optim as optim
from util import *

from ofa.nas.accuracy_predictor import *
from ofa.model_zoo import ofa_net
from ofa.imagenet_classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.imagenet_classification.run_manager import DistributedImageNetRunConfig
from ofa.utils import download_url, MyRandomResizedCrop
from ofa.utils import AverageMeter, cross_entropy_loss_with_soft_target
from ofa.utils import DistributedMetric, list_mean, subset_mean, val2list, MyRandomResizedCrop
from ofa.imagenet_classification.run_manager import DistributedRunManager
from ofa.nas.accuracy_predictor import AccuracyPredictor, ResNetArchEncoder


def test(val_loader, model, epoch, args, stats=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            #if args.gpu is not None:
            #    images = images.cuda(args.gpu, non_blocking=True)
            #target = target.cuda(args.gpu, non_blocking=True)
            images, target = images.cuda(), target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)


        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


    return top1.avg


def train(train_loader,optimizer, model, epoch, args, stats=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    model.train()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, target = images.cuda(), target.cuda()


        # compute output
        output= model(images)
        #rint('stats',stats)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            progress.display(i,optimizer)
    
    print('Finished Training')

    return



if __name__=='__main__':

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    #args.teacher_path = download_url(
    #    'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
    #    model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
    #)

    num_gpus = hvd.size()

    parser = argparse.ArgumentParser(description='Train acc_predictor for once-for-all')
    parser.add_argument('--epochs', type=int, default=450, metavar='N',
            help='number of epochs to train (default: 450)')
    parser.add_argument('--lr_epochs', type=int, default=100, metavar='N',
            help='number of epochs to change lr (default: 100)')
    parser.add_argument('--pretrained', default=None, nargs='+',
            help='pretrained model ( for mixtest \
            the first pretrained model is the big one \
            and the sencond is the small net)')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--gen_dataset', default=False, 
                    help='seed for initializing training. ')
    args = parser.parse_args()

    # https://github.com/mit-han-lab/once-for-all/issues/30

    args.path='./acc_dataset'

    args.manual_seed = 0

    args.lr_schedule_type = 'cosine'

    args.base_batch_size = 64
    args.valid_size = 10000

    args.opt_type = 'adam'
    args.base_lr = 1e-3
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.no_nesterov = False

    args.model_init = 'he_fout'
    args.validation_frequency = 1
    args.print_frequency = 10

    args.n_worker = 8
    args.resize_scale = 0.08
    args.distort_color = 'tf'
    args.image_size = '128, 144, 160, 176, 192, 224, 240, 256'
    args.continuous_size = True
    args.not_sync_distributed_image_size = False

    args.bn_momentum = 0.1
    args.bn_eps = 1e-5
    args.dropout = 0.1


    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4
    run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

    # print run config information
    if hvd.rank() == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    """ Distributed RunManager """
    ofa_network = ofa_net('ofa_resnet50_expand', pretrained=False)
    state_dict = torch.load('./exp/kernel_depth2expand/phase2/checkpoint/model_best.pth.tar')['state_dict']
    ofa_network.load_state_dict(state_dict)

    print('depth_list', ofa_network.depth_list)
    print('expand_ratio_list', ofa_network.expand_ratio_list)
    print('width_mult_list', ofa_network.width_mult_list)
    arch_encoder = ResNetArchEncoder(
        image_size_list=args.image_size, depth_list=ofa_network.depth_list, expand_list=ofa_network.expand_ratio_list,
        width_mult_list=ofa_network.width_mult_list, base_depth_list=ofa_network.BASE_DEPTH_LIST )

    compression = hvd.Compression.none
    run_manager = DistributedRunManager(
        args.path, ofa_network, run_config, compression, is_root=(hvd.rank() == 0)
    )
    run_manager.save_config()
    # hvd broadcast
    run_manager.broadcast()

    '''
    Build Accuracy Predictor Dataset
    '''
    acc_dataset = AccuracyDataset('./acc_dataset')
    if args.gen_dataset:
        acc_dataset.build_acc_dataset(run_manager, ofa_network)
        acc_dataset.merge_acc_dataset()

    acc_predictor_train_loader, acc_predictor_valid_loader, acc_predictor_base_acc = \
        acc_dataset.build_acc_data_loader(arch_encoder)
    
    '''
    Train Accuracy Predictor
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    acc_predictor = AccuracyPredictor(arch_encoder, 400, 3,
                                  checkpoint_path=None, device=device)
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(acc_predictor.parameters(), 
                lr=args.lr, weight_decay= args.weight_decay)
    
    for epoch in range(0,args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(acc_predictor_train_loader,optimizer, acc_predictor, epoch, args)
        acc = test(acc_predictor_valid_loader, acc_predictor, epoch, args)
        if (acc > bestacc):
            bestacc = acc
            save_state(acc_predictor,acc,epoch,args, optimizer, True)
        else:
            save_state(acc_predictor,bestacc,epoch,args,optimizer, False)
        print('best acc so far:{:4.2f}'.format(bestacc))
