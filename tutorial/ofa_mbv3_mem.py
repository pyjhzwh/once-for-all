from ofa.model_zoo import ofa_net
from torchvision import transforms, datasets
import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
#from matplotlib import pyplot as plt
#from ofa.nas.search_algorithm import EvolutionFinder
#from ofa.tutorial import AccuracyPredictor
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager



ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
# set random seed
random_seed = 2
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

# accuracy predictor
import torch
#from ofa.nas.accuracy_predictor import AccuracyPredictor, MobileNetArchEncoder
from ofa.tutorial import AccuracyPredictor, EvolutionFinder, FLOPsTable, LatencyTable
from ofa.utils import download_url

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
image_size_list = [128, 144, 160, 176, 192, 224, 240, 256]
'''
arch_encoder = MobileNetArchEncoder(
	image_size_list=image_size_list, ks_list=ofa_network.ks_list, depth_list=ofa_network.depth_list, 
    expand_list=ofa_network.expand_ratio_list
)


acc_predictor_checkpoint_path = download_url(
    'https://hanlab.mit.edu/files/OnceForAll/tutorial/acc_predictor.pth',
    model_dir='~/.ofa/',
)

acc_predictor = AccuracyPredictor(arch_encoder, 400, 3,
                                  checkpoint_path=acc_predictor_checkpoint_path, device=device)
'''
# accuracy predictor
acc_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)

print('The accuracy predictor is ready!')
print(acc_predictor.model)

#from ofa.nas.efficiency_predictor import Mbv3FLOPsModel

#efficiency_predictor = Mbv3FLOPsModel(ofa_network)
efficiency_predictor = FLOPsTable(
    device=device,
    batch_size=1,
)
print('The FLOPs lookup table is ready!')

from ofa.nas.memory_predictor import Mbv3WorkingMemModel 
memory_predictor_baseline = Mbv3WorkingMemModel(ofa_network, 0)
memory_predictor_ideal = Mbv3WorkingMemModel(ofa_network, 1)
memory_predictor_self = Mbv3WorkingMemModel(ofa_network, 2)

def build_val_transform(size):
    return transforms.Compose([
        transforms.Resize(int(math.ceil(size / 0.875))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
# path to the ImageNet dataset
imagenet_data_path = '/data2/jiecaoyu/imagenet/imgs/'
data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        root=os.path.join(imagenet_data_path, 'val'),
        transform=build_val_transform(224)
    ),
    batch_size=250,  # test batch size
    shuffle=True,
    num_workers=12,  # number of workers for the data loader
    pin_memory=True,
    drop_last=False,
)
print('The ImageNet dataloader is ready.')
run_config = ImagenetRunConfig(test_batch_size=100, n_worker=12)


""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
######################################
############ baseline
######################################
print('-'*50)
print('baseline NAS')
print('-'*50)
FLOPs_constraint = 5000  # MFLOPs
workingmem_constraint = 1024# KB
P = 100  # The size of population in each generation
N = 200  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    #'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    #'efficiency_constraint': FLOPs_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': efficiency_predictor, # To use a predefined efficiency predictor.
    'accuracy_predictor': acc_predictor, # To use a predefined accuracy_predictor predictor.
    'memory_predictor': memory_predictor_baseline, # To use a predefined working memory predictor
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# build the evolution finder
finder = EvolutionFinder(**params)

# start searching
result_lis = []
st = time.time()
best_valids, best_info = finder.run_evolution_search(FLOPs_constraint, workingmem_constraint, verbose=False)
acc, net_config, FLOPS, workingmem = best_info
sample_image_size = net_config['r'][0]
result_lis.append(best_info)
ed = time.time()
print('Found best architecture with FLOPS <= %.2f M and working mem <= %.2f in %.2f seconds! '
      'It achieves %.2f%s predicted accuracy ' %
      (FLOPs_constraint, workingmem_constraint, ed-st, acc * 100, '%'))

# visualize the architecture of the searched sub-net

ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
print('Architecture of the searched sub-net:')
print(ofa_network.module_str)
print('image size:', sample_image_size)
print('FLOPS',FLOPS,'workingmem', workingmem)

subnet = ofa_network.get_active_subnet(preserve_weight=True)
run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)
# assign image size: 128, 132, ..., 224
run_config.data_provider.assign_active_img_size(sample_image_size)
run_manager.reset_running_statistics(net=subnet)
loss, (top1, top5) = run_manager.validate(net=subnet)
print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))

