# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import module
import math

__all__ = ['AccuracyPredictor']


class AccuracyPredictor(nn.Module):

	def __init__(self, arch_encoder, hidden_size=400, n_layers=3,
	             checkpoint_path=None, device='cuda:0'):
		super(AccuracyPredictor, self).__init__()
		self.arch_encoder = arch_encoder
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.device = device

		# build layers
		layers = []
		for i in range(self.n_layers):
			layers.append(nn.Sequential(
				nn.Linear(self.arch_encoder.n_dim if i == 0 else self.hidden_size, self.hidden_size),
				nn.ReLU(inplace=True),
			))
		layers.append(nn.Linear(self.hidden_size, 1, bias=True))
		self.layers = nn.Sequential(*layers)
		self.base_acc = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=False)

		if checkpoint_path is not None and os.path.exists(checkpoint_path):
			checkpoint = torch.load(checkpoint_path, map_location='cpu')
			
			if 'state_dict' in checkpoint:
				checkpoint = checkpoint['state_dict']
			'''
			for key in list(checkpoint.keys()):
				if 'weight' in key or 'bias' in key:
					split_key = key.split('.')
					layer_num = int(int(split_key[0])/2)
					if layer_num == 3:
						new_key = 'layers.'+str(layer_num)+'.'+split_key[1]
					else:
						new_key = 'layers.'+str(layer_num)+'.0.'+split_key[1]
					checkpoint[new_key] = checkpoint.pop(key)
			'''
			self.load_state_dict(checkpoint,strict=False)
			print('Loaded checkpoint from %s' % checkpoint_path)

		else:
			for m in self.modules():
				if isinstance(m, nn.Conv2d):
					n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
					m.weight.data.normal_(0, math.sqrt(2. / n))
				elif isinstance(m, nn.BatchNorm2d):
					m.weight.data.fill_(1)
					m.bias.data.zero_()
					#nn.init.constant_(m.weight,1)
					#nn.init.constant_(m.bias,0)

		self.layers = self.layers.to(self.device)

	def forward(self, x):
		y = self.layers(x).squeeze()
		return y + self.base_acc

	def predict_acc(self, arch_dict_list):
		X = [self.arch_encoder.arch2feature(arch_dict) for arch_dict in arch_dict_list]
		X = torch.tensor(np.array(X)).float().to(self.device)
		return self.forward(X)
