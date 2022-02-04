# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse
import time
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.model_zoo import ofa_net
from util import *

def test(val_loader, model, args):
    batch_time = AverageMeterName('Time', ':6.3f')
    losses = AverageMeterName('Loss', ':.4e')
    top1 = AverageMeterName('Acc@1', ':6.2f')
    top5 = AverageMeterName('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        end = time.time()
        # apply quantized value to testing stage

        for i, (images, target) in enumerate(val_loader):
            #if args.gpu is not None:
            #    images = images.cuda(args.gpu, non_blocking=True)
            #target = target.cuda(args.gpu, non_blocking=True)
            images, target = images.cuda(), target.cuda()
            #print(images[0,0,0,:10])
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

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--path',
    help='The path of imagenet',
    type=str,
    default='/data2/jiecaoyu/imagenet/imgs/')
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='all')
parser.add_argument(
    '-b',
    '--batch-size',
    help='The batch on every device for validation',
    type=int,
    default=100)
parser.add_argument(
    '-j',
    '--workers',
    help='Number of workers',
    type=int,
    default=12)
parser.add_argument(
    '-n',
    '--net',
    metavar='OFANET',
    default='ofa_resnet50',
    choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e2346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2', 'ofa_proxyless_d234_e346_k357_w1.3',
             'ofa_resnet50', 'ofa_resnet50_expand','resnet50d'],
    help='OFA networks')

args = parser.parse_args()
if args.gpu == 'all':
    device_list = range(torch.cuda.device_count())
    args.gpu = ','.join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.path

ofa_network = ofa_net(args.net, pretrained=True)
#run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers)

nclass=1000
imagenet_datapath = args.path
traindir = os.path.join(imagenet_datapath,'train')
testdir = os.path.join(imagenet_datapath,'val')
#torchvision.set_image_backend('accimage')

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trainset = torchvision.datasets.ImageFolder(root=traindir,transform=
                                    transforms.Compose([
                                        #transforms.Resize(256),
                                        #transforms.CenterCrop(args.crop),
                                        #transforms.RandomCrop(args.crop),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                        ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                        shuffle=True, num_workers=12)

testset = torchvision.datasets.ImageFolder(root=testdir,transform=
                                    transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                        ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                        shuffle=False, num_workers=12)


if "ofa" in args.net:
    """ Set full network
    """
    ofa_network.set_max_net()
    arch_config = ofa_network.get_max_net_config()
    print('arch_config', arch_config)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
else:
    subnet = ofa_network
""" Test full network
"""
subnet.cuda()
criterion = nn.CrossEntropyLoss().cuda()
###### WARNING: NEED TO reset_running_statistics before test or the acc is wrong
test(testloader, subnet,args)