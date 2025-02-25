# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random

import horovod.torch as hvd
from numpy.core.fromnumeric import sort
import torch
from ofa.imagenet_classification.elastic_nn.networks.ofa_resnets import OFAResNets

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_classification.run_manager import DistributedImageNetRunConfig
from ofa.imagenet_classification.networks import ResNet50D
from ofa.imagenet_classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.utils import download_url, MyRandomResizedCrop
from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import load_models

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='depth', choices=[
    'depth', 'expand', 'width'
])
parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
parser.add_argument('--resume', action='store_true')

args = parser.parse_args()
if args.task == 'depth':
    # no need to add more flexbility to depth for 
    # model ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0
    raise NotImplementedError
    '''
    args.path = 'exp/normal2kernel_depth/phase%d' % args.phase
    args.dynamic_batch_size = 2
    # add one more depth option
    # '0,1,2' to '0,1,2,3'
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = '3'
        args.width_mult_list = '0.65,0.8,1.0'
        args.expand_list = '0.2,0.25,0.35'
        args.depth_list = '0,1,2,3'
    '''
elif args.task == 'expand':
    args.path = 'exp/expand/phase%d' % args.phase
    args.dynamic_batch_size = 2
    # from e=0.2,0.25,0.35 to e=0.15,0.2,0.25,0.35
    if args.phase == 1:
        args.n_epochs = 20
        args.base_lr = 1e-5 #1e-2
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = '3'
        args.width_mult_list = '0.65,0.8,1.0'
        args.expand_list = '0.15,0.2,0.25,0.35'
        args.depth_list = '0,1,2'
    # from e=0.15,0.2,0.25,0.35 to e=0.1,0.15,0.2,0.25,0.35
    else:
        args.n_epochs = 25
        args.base_lr = 3e-3
        args.warmup_epochs = 1
        args.warmup_lr = -1
        args.ks_list = '3'
        args.width_mult_list = '0.65,0.8,1.0'
        args.expand_list = '0.1,0.15,0.2,0.25,0.35'
        args.depth_list = '0,1,2'
elif args.task == 'width':
    args.path = 'exp/width_mult/phase%d' % args.phase
    args.dynamic_batch_size = 1
    # from w=0.65,0.8,1.0 to w=0.5,0.65,0.8,1.0
    if args.phase == 1:
        args.n_epochs = 20
        args.base_lr = 3e-3
        args.warmup_epochs = 1
        args.warmup_lr = -1
        args.ks_list = '3'
        args.width_mult_list = '0.5,0.65,0.8,1.0'
        args.expand_list = '0.2,0.25,0.35'
        args.depth_list = '0,1,2'
    # from w=0.5,0.65,0.8,1.0 to w=0,1,0.15,0.2,0.25,0.35
    elif args.phase == 2:
        args.n_epochs = 10
        args.base_lr = 4e-3
        args.warmup_epochs = 1
        args.warmup_lr = -1
        args.ks_list = '3'
        args.width_mult_list = '0.35,0.5,0.65,0.8,1.0'
        args.expand_list = '0.2,0.25,0.35'
        args.depth_list = '0,1,2'
else:
    raise NotImplementedError
args.manual_seed = 0

#args.lr_schedule_type = 'cosine'
args.lr_schedule_type = 'exp'

args.base_batch_size = 50
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

args.kd_ratio = 0.5
args.kd_type = 'ce'


if __name__ == '__main__':
    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    #args.teacher_path = download_url(
    #    'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
    #    model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
    #)
    args.teacher_path = '.torch/ofa_nets/resnet50d_ra2-464e36ba.pth'

    num_gpus = hvd.size()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

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
    
    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
   
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [float(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]
    

    args.width_mult_list = args.width_mult_list[0] if len(args.width_mult_list) == 1 else args.width_mult_list
    net = OFAResNets(
        n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
        width_mult_list=args.width_mult_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list
    )
    # teacher model
    if args.kd_ratio > 0:
        args.teacher_model = ResNet50D(
            n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=0, width_mult=1.0, expand_ratio=0.25, depth_param=None,
        )
        args.teacher_model.cuda()

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    distributed_run_manager = DistributedRunManager(
        args.path, net, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(hvd.rank() == 0)
    )
    distributed_run_manager.save_config()
    # hvd broadcast
    distributed_run_manager.broadcast()

    # load teacher net weights
    if args.kd_ratio > 0:
        load_models(distributed_run_manager, args.teacher_model, model_path=args.teacher_path)

    # training
    from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import validate, train

    validate_func_dict = {'image_size_list': {224} if isinstance(args.image_size, int) else sorted({160, 224}),
                          'ks_list': sorted({min(args.ks_list), max(args.ks_list)}),
                          'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
                          'depth_list': sorted({min(net.depth_list), max(net.depth_list)})} 
    # Note: do not specify 'width_mult_list' because it is index (int) for the args.width_mult_list
    if args.task == 'depth':
        raise NotImplementedError
        '''
        from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import train_elastic_depth
        validate_func_dict['depth_list'] = sorted(args.depth_list)
        if distributed_run_manager.start_epoch == 0:
            args.ofa_checkpoint_path = download_url(
                'https://hanlab.mit.edu/files/OnceForAll/ofa_nets/ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0',
                model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
            )
            load_models(distributed_run_manager, distributed_run_manager.net, args.ofa_checkpoint_path)
            distributed_run_manager.write_log(
                '%.3f\t%.3f\t%.3f\t%s' % validate(distributed_run_manager, is_test=True, **validate_func_dict), 'valid')
        else:
            assert args.resume
        train(distributed_run_manager, args,
              lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
        '''
    elif args.task == 'expand':
        from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import train_elastic_expand
        if args.phase == 1:
            args.ofa_checkpoint_path = download_url(
                'https://hanlab.mit.edu/files/OnceForAll/ofa_nets/ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0',
                model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
            )
            args.ofa_checkpoint_path = 'back/back3/phase1/checkpoint/model_best.pth.tar'
            args.resume = True
        else:
            args.ofa_checkpoint_path = 'exp/kernel_depth2expand/phase1/checkpoint/model_best.pth.tar'
        train_elastic_expand(train, distributed_run_manager, args, validate_func_dict)
    elif args.task == 'width':
        from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import train_elastic_width_mult
        if args.phase == 1:
            #args.ofa_checkpoint_path = 'exp/kernel_depth2expand/phase2/checkpoint/model_best.pth.tar'
            args.ofa_checkpoint_path = download_url(
                'https://hanlab.mit.edu/files/OnceForAll/ofa_nets/ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0',
                model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
            )
        else:
            args.ofa_checkpoint_path = 'exp/expand2width_mult/phase1/checkpoint/model_best.pth.tar'
        train_elastic_width_mult(train, distributed_run_manager, args, validate_func_dict)
    else:
        raise NotImplementedError
