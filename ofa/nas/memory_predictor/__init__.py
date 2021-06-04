# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import copy
from .mem_predictor import *
#from .self_conv_mem import *


class BaseWorkingMemModel:

	def __init__(self, ofa_net, type=0):
		self.ofa_net = ofa_net
		# 0 baseline mem count; 1: self-loop mem count
		self.type = type

	def get_active_subnet_config(self, arch_dict):
		arch_dict = copy.deepcopy(arch_dict)
		image_size = arch_dict.pop('image_size')
		self.ofa_net.set_active_subnet(**arch_dict)
		active_net_config = self.ofa_net.get_active_net_config()
		return active_net_config, image_size

	def get_workingmem(self, arch_dict):
		raise NotImplementedError


class ProxylessNASWorkingMemModel(BaseWorkingMemModel):

	def get_workingmem(self, arch_dict):
		active_net_config, image_size = self.get_active_subnet_config(arch_dict)
		return ProxylessNASWorkingMemTable.count_workingmem_given_config(active_net_config, image_size)


class Mbv3WorkingMemModel(BaseWorkingMemModel):

	def get_workingmem(self, arch_dict):
		active_net_config, image_size = self.get_active_subnet_config(arch_dict)
		return MBv3WorkingMemTable.count_workingmem_given_config(active_net_config, image_size)


class ResNet50WorkingMemModel(BaseWorkingMemModel):
	def get_workingmem(self, arch_dict):
		active_net_config, image_size = self.get_active_subnet_config(arch_dict)
		return ResNet50WorkingMemTable.count_workingmem_given_config(active_net_config, image_size, self.type)


