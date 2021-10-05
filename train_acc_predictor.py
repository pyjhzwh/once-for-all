from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import horovod.torch as hvd

from ofa.nas.accuracy_predictor import *
from ofa.model_zoo import ofa_net
from ofa.imagenet_classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.imagenet_classification.run_manager import DistributedImageNetRunConfig
from ofa.utils import download_url, MyRandomResizedCrop



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
    parser.add_argument('--no_cuda', default=False, 
            help = 'do not use cuda',action='store_true')
    parser.add_argument('--epochs', type=int, default=450, metavar='N',
            help='number of epochs to train (default: 450)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr_epochs', type=int, default=100, metavar='N',
            help='number of epochs to change lr (default: 100)')
    parser.add_argument('--pretrained', default=None, nargs='+',
            help='pretrained model ( for mixtest \
            the first pretrained model is the big one \
            and the sencond is the small net)')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    default=False, help='evaluate model on validation set')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--arch', action='store', default='resnet20',
                        help='the CIFAR10 network structure: \
                        resnet20 | resnet18 | resnet50 | all_cnn_net | alexnet')
    parser.add_argument('--dataset', action='store', default='cifar10',
            help='pretrained model: cifar10 | imagenet')
    args = parser.parse_args()

    args.manual_seed = 0

    args.lr_schedule_type = 'cosine'

    args.base_batch_size = 32
    args.valid_size = 10000

    args.opt_type = 'sgd'
    args.momentum = 0.9
    args.no_nesterov = False
    args.weight_decay = 3e-5
    args.label_smoothing = 0.1
    args.no_decay_keys = 'bn#bias'
    args.fp16_allreduce = False

    args.model_init = 'he_fout'
    args.validation_frequency = 1
    args.print_frequency = 10

    args.n_worker = 8
    args.resize_scale = 0.08
    args.distort_color = 'tf'
    args.image_size = '128,160,192,224'
    args.continuous_size = True
    args.not_sync_distributed_image_size = False

    args.bn_momentum = 0.1
    args.bn_eps = 1e-5
    args.dropout = 0.1
    args.base_stage_width = 'proxyless'

    args.dy_conv_scaling_mode = 1
    args.independent_distributed_sampling = False

    args.kd_ratio = 0
    args.kd_type = 'ce'

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
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4
    run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

    # print run config information
    if hvd.rank() == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.

    ofa_network = ofa_net('ofa_resnet50_expand', pretrained=False)
    state_dict = torch.load('../exp/kernel_depth2expand/phase2/checkpoint/model_best.pth.tar')['state_dict']
    ofa_network.load_state_dict(state_dict)

    arch_encoder = ResNetArchEncoder(
        image_size_list=args.image_size_list, depth_list=ofa_network.depth_list, expand_list=ofa_network.expand_ratio_list,
        width_mult_list=ofa_network.width_mult_list, base_depth_list=ofa_network.BASE_DEPTH_LIST )

    compression = hvd.Compression.none
    run_manager = DistributedRunManager(
        args.path, ofa_network, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(hvd.rank() == 0)
    )
    run_manager.save_config()
    # hvd broadcast
    run_manager.broadcast()

    '''
    Build Accuracy Predictor Dataset
    '''
    acc_dataset = AccuracyDataset('./acc_dataset')
    acc_dataset.build_acc_dataset(run_manager, ofa_network)
    acc_dataset.merge_acc_dataset()
    acc_predictor_train_loader, acc_predictor_valid_loader, acc_predictor_base_acc = \
        acc_dataset.build_acc_data_loader(arch_encoder)
    
    '''
    train Accuracy Predictor
    '''
    distributed = isinstance(run_manager, DistributedRunManager)

    for epoch in range(run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs):
        train_loss, (train_top1, train_top5) = train(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr)

        if (epoch + 1) % args.validation_frequency == 0:
            val_loss, val_acc, val_acc5, _val_log = test(run_manager, epoch=epoch, is_test=False)
            # best_acc
            is_best = val_acc > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, val_acc)
            if not distributed or run_manager.is_root:
                val_log = 'Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - args.warmup_epochs, run_manager.run_config.n_epochs, val_loss, val_acc,
                           run_manager.best_acc)
                val_log += ', Train top-1 {top1:.3f}, Train loss {loss:.3f}\t'.format(top1=train_top1, loss=train_loss)
                val_log += _val_log
                run_manager.write_log(val_log, 'valid', should_print=False)

                run_manager.save_model({
                    'epoch': epoch,
                    'best_acc': run_manager.best_acc,
                    'optimizer': run_manager.optimizer.state_dict(),
                    'state_dict': run_manager.network.state_dict(),
                }, is_best=is_best)

