# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from re import M
from typing import List
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer, DynamicConv2d, DynamicSeparableConv2d
from ofa.utils.layers import ConvLayer
from torch.nn.modules.conv import Conv2d
from torchvision.transforms.functional import pad
from ofa.utils import download_url, make_divisible, MyNetwork
#from utils.layers import MBConvLayer
from .self_conv_mem import *


__all__ = ['count_ideal_conv_mem', 'count_baseline_conv_mem', 'ProxylessNASWorkingMemTable', 'MBv3WorkingMemTable', 'ResNet50WorkingMemTable']


def count_ideal_conv_mem(in_size, out_size, in_channels, out_channels):
	working_mem = max(in_size * in_size * in_channels, out_size * out_size * out_channels)
	return working_mem

def count_baseline_conv_mem(in_size, out_size, in_channels, out_channels):
	working_mem = in_size * in_size * in_channels + out_size * out_size * out_channels
	return working_mem

def count_selfloop_conv_mem(conv_layer_param):
	mem_planner = MemoryAllocation(conv_layer_param)
	return mem_planner.actual_mem_size()

def count_pool_mem(in_size, layer_config, channel, type=0):
	padding = layer_config['padding']
	kernel_size = layer_config['kernel_size']
	stride = layer_config['stride']
	out_size = int((in_size + 2 * padding - kernel_size) / stride + 1)
	
	if type == 0:
		workingmem = count_baseline_conv_mem(in_size, out_size,channel,channel)
	elif type == 1:
		workingmem = count_ideal_conv_mem(in_size, out_size,channel,channel)
	elif type == 2:
		layer = Conv_layer_param('input_stem', in_size, in_size, channel, kernel_size,
				padding, stride, out_size, out_size, channel)
		workingmem = count_selfloop_conv_mem(layer)
	return workingmem, out_size

def count_conv_mem(in_size, layer_config, type=0):
	#layer_config = net_config['first_conv']['first_conv']
	in_channel = layer_config['in_channels']
	out_channel = layer_config['out_channels']
	padding = layer_config['padding']
	dilation = layer_config['dilation']
	kernel_size = layer_config['kernel_size']
	stride = layer_config['stride']
	if isinstance(padding, tuple):
		padding = padding[0]
	if isinstance(dilation, tuple):
		dilation = dilation[0] 
	if isinstance(kernel_size, tuple):
		kernel_size = kernel_size[0]
	if isinstance(stride, tuple):
		stride = stride[0] 
	out_size = int((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
	
	if type == 0:
		workingmem = count_baseline_conv_mem(in_size, out_size,in_channel,out_channel)
	elif type == 1:
		#print(in_size, out_size,in_channel,out_channel, count_ideal_conv_mem(in_size, out_size,in_channel,out_channel))
		workingmem = count_ideal_conv_mem(in_size, out_size,in_channel,out_channel)
	elif type == 2:
		layer = Conv_layer_param('input_stem', in_size, in_size, in_channel, layer_config['kernel_size'],
				layer_config['padding'], layer_config['stride'], out_size, out_size, out_channel)
		workingmem = count_selfloop_conv_mem(layer)
	return workingmem, out_size

def count_depth_conv_mem(in_size, layer_config, type=0):
	#layer_config = net_config['first_conv']['first_conv']
	in_channel = layer_config['in_channels']
	out_channel = layer_config['out_channels']
	padding = layer_config['padding']
	dilation = layer_config['dilation']
	kernel_size = layer_config['kernel_size']
	stride = layer_config['stride']
	if isinstance(padding, tuple):
		padding = padding[0]
	if isinstance(dilation, tuple):
		dilation = dilation[0] 
	if isinstance(kernel_size, tuple):
		kernel_size = kernel_size[0]
	if isinstance(stride, tuple):
		stride = stride[0] 
	out_size = int((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)	
	if type == 0:
		workingmem = count_baseline_conv_mem(in_size, out_size,in_channel,out_channel)
	elif type == 1:
		#print(in_size, out_size,in_channel,out_channel, count_ideal_conv_mem(in_size, out_size,in_channel,out_channel))
		workingmem = count_ideal_conv_mem(in_size, out_size,in_channel,out_channel)
	elif type == 2:
		workingmem = count_ideal_conv_mem(in_size, out_size,in_channel,out_channel)
	return workingmem, out_size

def build_layer_config_from_conv(conv):
	if isinstance(conv, Conv2d) or isinstance(conv, ConvLayer):
		layer_config = {
			'in_channels' : conv.in_channels,
			'out_channels': conv.out_channels,
			'kernel_size': conv.kernel_size,
			'stride': conv.stride,
			'padding': conv.padding,
			'dilation': conv.dilation
		}
	elif isinstance(conv, DynamicConv2d):
		layer_config = {
			'in_channels' : conv.max_in_channels,
			'out_channels': conv.active_out_channel,
			'kernel_size': conv.kernel_size,
			'stride': conv.stride,
			'padding': conv.padding,
			'dilation': conv.dilation
		}
	elif isinstance(conv, DynamicSeparableConv2d):
		layer_config = {
			'in_channels' : conv.max_in_channels,
			'out_channels': conv.max_in_channels,
			'kernel_size': conv.active_kernel_size,
			'stride': conv.stride,
			'padding': 0,
			'dilation': conv.dilation
		}
	return layer_config


class MemLog():

	def __init__(self) -> None:
		self.log = {}

	def set_image_size(self, image_size) -> None:
		self.image_size = image_size
	
	def log_mem(self, key, mem) ->None:
		if key in self.log:
			if 'mem' in self.log[key]:
				raise ValueError(key+'already exists in dict, use a different name')
			self.log[key]['mem'] = mem
		else:
			self.log[key] = {'mem': mem}

	def log_config(self, key, config) ->None:
		if key in self.log:
			if 'config' in self.log[key]:
				raise ValueError(key+'already exists in dict, use a different name')
			self.log[key]['config'] = config
		else:
			self.log[key] = {'config': config}

	def add_log(self, key, config, mem) -> None:
		if key in self.log:
			raise ValueError(key+'already exists in dict, use a different name')
		else:
			self.log[key] = {'config': config, 'mem':mem}


	def peak_mem(self) -> Any:
		max_mem = 0
		for k,v in self.log.items():
			if v['mem'] >= max_mem:
				max_mem = v['mem']

		return max_mem

	def peak_mem_usage(self) -> List:
		layers = []
		max_mem = self.peak_mem()
		for k,v in self.log.items():
			if v['mem'] == max_mem:
				layers.append(k)

		return layers
	
	def print(self):
		#import pprint
		if self.image_size:
			print('image_size', self.image_size)
		#pprint.pprint(self.log)
		for k,v in self.log.items():
			print(k,v)
		print('peak_memory', self.peak_mem())
		print('peak_memory layers', self.peak_mem_usage())

class WorkingMemTable(object):


	@staticmethod
	def repr_shape(shape):
		if isinstance(shape, (list, tuple)):
			return 'x'.join(str(_) for _ in shape)
		elif isinstance(shape, str):
			return shape
		else:
			return TypeError

	def query(self, **kwargs):
		raise NotImplementedError

	def predict_network_latency(self, net, image_size):
		raise NotImplementedError

	def predict_network_latency_given_config(self, net_config, image_size):
		raise NotImplementedError

	@staticmethod
	def count_workingmem_given_config(net_config, image_size=224):
		raise NotImplementedError


class ProxylessNASWorkingMemTable(WorkingMemTable):

	def query(self, l_type: str, input_shape, output_shape, expand=None, ks=None, stride=None, id_skip=None):
		"""
		:param l_type:
			Layer type must be one of the followings
			1. `Conv`: The initial 3x3 conv with stride 2.
			2. `Conv_1`: feature_mix_layer
			3. `Logits`: All operations after `Conv_1`.
			4. `expanded_conv`: MobileInvertedResidual
		:param input_shape: input shape (h, w, #channels)
		:param output_shape: output shape (h, w, #channels)
		:param expand: expansion ratio
		:param ks: kernel size
		:param stride:
		:param id_skip: indicate whether has the residual connection
		"""
		infos = [l_type, 'input:%s' % self.repr_shape(input_shape), 'output:%s' % self.repr_shape(output_shape), ]

		if l_type in ('expanded_conv',):
			assert None not in (expand, ks, stride, id_skip)
			infos += ['expand:%d' % expand, 'kernel:%d' % ks, 'stride:%d' % stride, 'idskip:%d' % id_skip]
		key = '-'.join(infos)
		return self.lut[key]['mean']

	def predict_network_latency(self, net, image_size=224):
		predicted_latency = 0
		# first conv
		predicted_latency += self.query(
			'Conv', [image_size, image_size, 3],
			[(image_size + 1) // 2, (image_size + 1) // 2, net.first_conv.out_channels]
		)
		# blocks
		fsize = (image_size + 1) // 2
		for block in net.blocks:
			mb_conv = block.conv
			shortcut = block.shortcut

			if mb_conv is None:
				continue
			if shortcut is None:
				idskip = 0
			else:
				idskip = 1
			out_fz = int((fsize - 1) / mb_conv.stride + 1)  # fsize // mb_conv.stride
			block_latency = self.query(
				'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
				expand=mb_conv.expand_ratio, ks=mb_conv.kernel_size, stride=mb_conv.stride, id_skip=idskip
			)
			predicted_latency += block_latency
			fsize = out_fz
		# feature mix layer
		predicted_latency += self.query(
			'Conv_1', [fsize, fsize, net.feature_mix_layer.in_channels],
			[fsize, fsize, net.feature_mix_layer.out_channels]
		)
		# classifier
		predicted_latency += self.query(
			'Logits', [fsize, fsize, net.classifier.in_features], [net.classifier.out_features]  # 1000
		)
		return predicted_latency

	def predict_network_latency_given_config(self, net_config, image_size=224):
		predicted_latency = 0
		# first conv
		predicted_latency += self.query(
			'Conv', [image_size, image_size, 3],
			[(image_size + 1) // 2, (image_size + 1) // 2, net_config['first_conv']['out_channels']]
		)
		# blocks
		fsize = (image_size + 1) // 2
		for block in net_config['blocks']:
			mb_conv = block['mobile_inverted_conv'] if 'mobile_inverted_conv' in block else block['conv']
			shortcut = block['shortcut']

			if mb_conv is None:
				continue
			if shortcut is None:
				idskip = 0
			else:
				idskip = 1
			out_fz = int((fsize - 1) / mb_conv['stride'] + 1)
			block_latency = self.query(
				'expanded_conv', [fsize, fsize, mb_conv['in_channels']], [out_fz, out_fz, mb_conv['out_channels']],
				expand=mb_conv['expand_ratio'], ks=mb_conv['kernel_size'], stride=mb_conv['stride'], id_skip=idskip
			)
			predicted_latency += block_latency
			fsize = out_fz
		# feature mix layer
		predicted_latency += self.query(
			'Conv_1', [fsize, fsize, net_config['feature_mix_layer']['in_channels']],
			[fsize, fsize, net_config['feature_mix_layer']['out_channels']]
		)
		# classifier
		predicted_latency += self.query(
			'Logits', [fsize, fsize, net_config['classifier']['in_features']],
			[net_config['classifier']['out_features']]  # 1000
		)
		return predicted_latency

	@staticmethod
	def count_workingmem_given_config(net_config, image_size=224):
		workingmem = 0
		# first conv
		workingmem += count_conv_mem((image_size + 1) // 2, 3, net_config['first_conv']['out_channels'], 3, 1)
		# blocks
		fsize = (image_size + 1) // 2
		for block in net_config['blocks']:
			mb_conv = block['mobile_inverted_conv'] if 'mobile_inverted_conv' in block else block['conv']
			if mb_conv is None:
				continue
			out_fz = int((fsize - 1) / mb_conv['stride'] + 1)
			if mb_conv['mid_channels'] is None:
				mb_conv['mid_channels'] = round(mb_conv['in_channels'] * mb_conv['expand_ratio'])
			if mb_conv['expand_ratio'] != 1:
				# inverted bottleneck
				workingmem += count_conv_mem(fsize, mb_conv['in_channels'], mb_conv['mid_channels'], 1, 1)
			# depth conv
			workingmem += count_conv_mem(out_fz, mb_conv['mid_channels'], mb_conv['mid_channels'],
									 mb_conv['kernel_size'], mb_conv['mid_channels'])
			# point linear
			workingmem += count_conv_mem(out_fz, mb_conv['mid_channels'], mb_conv['out_channels'], 1, 1)
			fsize = out_fz
		# feature mix layer
		workingmem += count_conv_mem(fsize, net_config['feature_mix_layer']['in_channels'],
								 net_config['feature_mix_layer']['out_channels'], 1, 1)
		# classifier
		workingmem += count_conv_mem(1, net_config['classifier']['in_features'],
								 net_config['classifier']['out_features'], 1, 1)
		return workingmem / 1e6  # Mmems


class MBv3WorkingMemTable(WorkingMemTable):

	def query(self, l_type: str, input_shape, output_shape, mid=None, ks=None, stride=None, id_skip=None,
			  se=None, h_swish=None):
		infos = [l_type, 'input:%s' % self.repr_shape(input_shape), 'output:%s' % self.repr_shape(output_shape), ]

		if l_type in ('expanded_conv',):
			assert None not in (mid, ks, stride, id_skip, se, h_swish)
			infos += ['expand:%d' % mid, 'kernel:%d' % ks, 'stride:%d' % stride, 'idskip:%d' % id_skip,
					  'se:%d' % se, 'hs:%d' % h_swish]
		key = '-'.join(infos)
		return self.lut[key]['mean']

	def predict_network_latency(self, net, image_size=224):
		predicted_latency = 0
		# first conv
		predicted_latency += self.query(
			'Conv', [image_size, image_size, 3],
			[(image_size + 1) // 2, (image_size + 1) // 2, net.first_conv.out_channels]
		)
		# blocks
		fsize = (image_size + 1) // 2
		for block in net.blocks:
			mb_conv = block.conv
			shortcut = block.shortcut

			if mb_conv is None:
				continue
			if shortcut is None:
				idskip = 0
			else:
				idskip = 1
			out_fz = int((fsize - 1) / mb_conv.stride + 1)
			block_latency = self.query(
				'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
				mid=mb_conv.depth_conv.conv.in_channels, ks=mb_conv.kernel_size, stride=mb_conv.stride, id_skip=idskip,
				se=1 if mb_conv.use_se else 0, h_swish=1 if mb_conv.act_func == 'h_swish' else 0,
			)
			predicted_latency += block_latency
			fsize = out_fz
		# final expand layer
		predicted_latency += self.query(
			'Conv_1', [fsize, fsize, net.final_expand_layer.in_channels],
			[fsize, fsize, net.final_expand_layer.out_channels],
		)
		# global average pooling
		predicted_latency += self.query(
			'AvgPool2D', [fsize, fsize, net.final_expand_layer.out_channels],
			[1, 1, net.final_expand_layer.out_channels],
		)
		# feature mix layer
		predicted_latency += self.query(
			'Conv_2', [1, 1, net.feature_mix_layer.in_channels],
			[1, 1, net.feature_mix_layer.out_channels]
		)
		# classifier
		predicted_latency += self.query(
			'Logits', [1, 1, net.classifier.in_features], [net.classifier.out_features]
		)
		return predicted_latency

	def predict_network_latency_given_config(self, net_config, image_size=224):
		predicted_latency = 0
		# first conv
		predicted_latency += self.query(
			'Conv', [image_size, image_size, 3],
			[(image_size + 1) // 2, (image_size + 1) // 2, net_config['first_conv']['out_channels']]
		)
		# blocks
		fsize = (image_size + 1) // 2
		for block in net_config['blocks']:
			mb_conv = block['mobile_inverted_conv'] if 'mobile_inverted_conv' in block else block['conv']
			shortcut = block['shortcut']

			if mb_conv is None:
				continue
			if shortcut is None:
				idskip = 0
			else:
				idskip = 1
			out_fz = int((fsize - 1) / mb_conv['stride'] + 1)
			if mb_conv['mid_channels'] is None:
				mb_conv['mid_channels'] = round(mb_conv['in_channels'] * mb_conv['expand_ratio'])
			block_latency = self.query(
				'expanded_conv', [fsize, fsize, mb_conv['in_channels']], [out_fz, out_fz, mb_conv['out_channels']],
				mid=mb_conv['mid_channels'], ks=mb_conv['kernel_size'], stride=mb_conv['stride'], id_skip=idskip,
				se=1 if mb_conv['use_se'] else 0, h_swish=1 if mb_conv['act_func'] == 'h_swish' else 0,
			)
			predicted_latency += block_latency
			fsize = out_fz
		# final expand layer
		predicted_latency += self.query(
			'Conv_1', [fsize, fsize, net_config['final_expand_layer']['in_channels']],
			[fsize, fsize, net_config['final_expand_layer']['out_channels']],
		)
		# global average pooling
		predicted_latency += self.query(
			'AvgPool2D', [fsize, fsize, net_config['final_expand_layer']['out_channels']],
			[1, 1, net_config['final_expand_layer']['out_channels']],
		)
		# feature mix layer
		predicted_latency += self.query(
			'Conv_2', [1, 1, net_config['feature_mix_layer']['in_channels']],
			[1, 1, net_config['feature_mix_layer']['out_channels']]
		)
		# classifier
		predicted_latency += self.query(
			'Logits', [1, 1, net_config['classifier']['in_features']], [net_config['classifier']['out_features']]
		)
		return predicted_latency

	@staticmethod
	def count_workingmem_given_net(net, image_size=224, type=0):
		workingmem = 0
		# first conv
		layer_config = net.config['first_conv']
		#print(net.module_str)
		workingmem = count_conv_mem(workingmem, image_size, layer_config, type)
		#print('image size', image_size)
		#print('workingmem', workingmem)
		# blocks
		fsize = (image_size + 1) // 2
		for block in net.blocks:
			mb_conv = block.conv
			if mb_conv is None:
				continue
			shortcut = block.shortcut
			prev_fsize = fsize
			#if isinstance(mb_conv, MBConvLayer) or isinstance(mb_conv, DynamicMBConvLayer):
			# inverted_bottlenect
			#print(mb_conv)
			if mb_conv.inverted_bottleneck:
				layer_config = build_layer_config_from_conv(mb_conv.inverted_bottleneck.conv)
				workingmem = count_conv_mem(workingmem, fsize, layer_config, type)
				#print('workingmem', workingmem, layer_config, 'fsize', fsize)
				stride = mb_conv.inverted_bottleneck.conv.stride
				if isinstance(stride, tuple):
					stride = stride[0]
				fsize = int((fsize - 1) / stride + 1)
			# depth_conv
			layer_config = build_layer_config_from_conv(mb_conv.depth_conv.conv)
			workingmem = count_depth_conv_mem(workingmem, fsize, layer_config, type)
			stride = mb_conv.depth_conv.conv.stride
			if isinstance(stride, tuple):
				stride = stride[0]
			fsize = int((fsize - 1) / stride + 1)
			#print('workingmem', workingmem, layer_config, 'fsize', fsize)
			# point_linear
			layer_config = build_layer_config_from_conv(mb_conv.point_linear.conv)
			workingmem = count_conv_mem(workingmem, fsize, layer_config, type)
			stride = mb_conv.point_linear.conv.stride
			if isinstance(stride, tuple):
				stride = stride[0]
			fsize = int((fsize - 1) / stride + 1)
			#print('workingmem', workingmem, layer_config, 'fsize', fsize)
			# residual layer
			if shortcut is not None and type != 0:
				workingmem += prev_fsize * prev_fsize * shortcut.in_channels
				#print('workingmem', workingmem, layer_config, 'fsize', fsize)
		# final expand layer
		layer_config = build_layer_config_from_conv(net.final_expand_layer)
		workingmem = count_conv_mem(workingmem, fsize, layer_config, type)
		fsize /= 6
		# feature mix layer
		layer_config = build_layer_config_from_conv(net.feature_mix_layer)
		workingmem = count_conv_mem(workingmem, fsize, layer_config, type)
		# classifier
		#workingmem += count_conv_mem(1, net.config['classifier']['in_features'],
		#						 net.config['classifier']['out_features'], 1, 1)
		#if(workingmem == 676*1024):
		#	exit()
		#print('-'*30)
		return workingmem / 1024  # KB

	@staticmethod
	def count_workingmem_given_config(net_config, image_size=224, type=0):
		workingmem = 0
		# first conv
		layer_config = net_config['first_conv']
		#print(net.module_str)
		#print('image size', image_size)
		curmem, fsize = count_conv_mem(image_size, layer_config, type)
		workingmem = max(workingmem, curmem)
		#print('curmem', curmem, 'fsize', fsize)
		# blocks
		#fsize = (image_size + 1) // 2
		for block in net_config['blocks']:
			mb_conv = block['conv']
			if mb_conv is None:
				continue
			shortcut = block['shortcut']
			prev_fsize = fsize
			#if isinstance(mb_conv, MBConvLayer) or isinstance(mb_conv, DynamicMBConvLayer):
			curblockmem  = 0
			# inverted_bottlenect
			if mb_conv['inverted_bottleneck']:
				layer_config = mb_conv['inverted_bottleneck']
				curmem, fsize = count_conv_mem(fsize, layer_config, type)
				curblockmem = max(curmem, curblockmem)
				#print('curmem', curmem, layer_config, 'fsize', fsize)
				stride = layer_config['stride']
				if isinstance(stride, tuple):
					stride = stride[0]
				#fsize = int((fsize - 1) / stride + 1)
			# depth_conv
			layer_config = mb_conv['depth_conv']
			curmem, fsize = count_depth_conv_mem(fsize, layer_config, type)
			curblockmem = max(curmem, curblockmem)
			stride = layer_config['stride']
			if isinstance(stride, tuple):
				stride = stride[0]
			#print('curmem', curmem, layer_config, 'fsize', fsize)
			# point_linear
			layer_config = mb_conv['point_linear']
			curmem, fsize = count_conv_mem(fsize, layer_config, type)
			curblockmem = max(curmem, curblockmem)
			stride = layer_config['stride']
			if isinstance(stride, tuple):
				stride = stride[0]
			#print('curmem', curmem, layer_config, 'fsize', fsize)
			# residual layer
			if shortcut is not None and type != 0:
				curblockmem += prev_fsize * prev_fsize * shortcut['in_channels']
				#print('curblockmem', curblockmem, layer_config, 'fsize', fsize)
			workingmem = max(workingmem, curblockmem)
		# final expand layer
		layer_config = net_config['final_expand_layer']
		curmem, fsize = count_conv_mem(fsize, layer_config, type)
		workingmem = max(curmem, workingmem)
		# pooling
		fsize /= 6
		# feature mix layer
		layer_config = net_config['feature_mix_layer']
		curmem, fsize = count_conv_mem(fsize, layer_config, type)
		workingmem = max(curmem, workingmem)
		#print('workingmem', workingmem)
		# classifier
		#workingmem += count_conv_mem(1, net.config['classifier']['in_features'],
		#						 net.config['classifier']['out_features'], 1, 1)
		#print('-'*30)
		return workingmem / 1024  # KB
		

class ResNet50WorkingMemTable(WorkingMemTable):

	def query(self, **kwargs):
		raise NotImplementedError

	def predict_network_latency(self, net, image_size):
		raise NotImplementedError

	def predict_network_latency_given_config(self, net_config, image_size):
		raise NotImplementedError

	@staticmethod
	def count_workingmem_given_config(net_config, image_size=224, type=0):
		# 0 baseline mem count; 1: ideal mem count; 2: self-loop mem count
		workingmem = 0
		mem_log = MemLog()
		mem_log.set_image_size(image_size)
		#tmp_workingmem = 0
		# input stem
		for i, layer_config in enumerate(net_config['input_stem']):
			#print(layer_config)
			key = 'input_stem'+'-'+str(i)+'-'+layer_config['name']
			if layer_config['name'] == 'ConvLayer':
				curmem, fsize = count_conv_mem(image_size, layer_config, type)
				workingmem = max(workingmem, curmem)
				mem_log.add_log(key, layer_config, curmem)
				channel = layer_config['out_channels']
			elif layer_config['name'] == 'ResidualBlock':
				conv_layer_config = layer_config['conv']
				curblockmem, fsize = count_conv_mem(image_size, conv_layer_config, type)
				mem_log.add_log(key+'conv', conv_layer_config, curblockmem)
				shortcut_config = layer_config['shortcut']
				if shortcut_config:
					curblockmem += image_size * image_size * conv_layer_config['out_channels']
					mem_log.add_log(key+'shortcut', shortcut_config, curblockmem)
				workingmem = max(workingmem, curblockmem)
				channel = conv_layer_config['out_channels']
			else:
				raise ValueError('layer type should be DynamicConvLayer or ResidualBlock')
			image_size = fsize
		# max pooling
		#print(net_config['blocks'])
		max_pool_mem, fsize = count_pool_mem(image_size, net_config['max_pooling'], channel, type)
		mem_log.add_log('max_pooling',net_config['max_pooling'], max_pool_mem)
		workingmem = max(workingmem, max_pool_mem)
		for i, block_config in enumerate(net_config['blocks']):
			key = 'block-'+str(i)+'-'+block_config['name']
			image_size = fsize
			curblockmem = 0
			conv1_config = block_config['conv1']
			conv2_config = block_config['conv2']
			conv3_config = block_config['conv3']
			curmem, fsize = count_conv_mem(fsize, conv1_config, type)
			mem_log.add_log(key+'-conv1', conv1_config, curmem)
			curblockmem = max(curmem, curblockmem)
			curmem, fsize = count_conv_mem(fsize, conv2_config, type)
			mem_log.add_log(key+'-conv2', conv2_config, curmem)
			curblockmem = max(curmem, curblockmem)
			curmem, fsize = count_conv_mem(fsize, conv3_config, type)
			mem_log.add_log(key+'-conv3', conv3_config, curmem)
			curblockmem = max(curmem, curblockmem)
			# downsample
			residual_size = 0
			#if block_config['stride'] == 1 and block_config['in_channel_list'] == block_config['out_channel_list']:
			# identity layer
			if block_config['downsample'] is None:
				residual_size = image_size * image_size * block_config['in_channels']
				residual_config = {'name', 'IdentityLayer'}
			elif block_config['downsample_mode'] == 'conv':
				residual_size,_ = count_conv_mem(image_size, block_config['downsample'], type)
				residual_config =  block_config['downsample']
			elif block_config['downsample_mode'] == 'avgpool_conv':
				# avg_pool
				image_size = (image_size + 1) //  block_config['stride']
				# conv
				residual_size,_ = count_conv_mem(image_size, block_config['downsample'], type)
				residual_config =  block_config['downsample']
			if residual_config:
				mem_log.add_log(key+'-downsample', residual_config, residual_size)
			curblockmem += residual_size
			workingmem = max(curblockmem, workingmem)	 

		# final classifier, fc
		#classifier_config = net_config['classifier']
		#curmem, fsize = count_conv_mem(fsize, classifier_config, type)
		#mem_log.add_log('classifier', curmem)
		#workingmem = max(curmem, workingmem)

		#mem_log.print()
		#exit()
		return workingmem / 1024, mem_log  # KB
