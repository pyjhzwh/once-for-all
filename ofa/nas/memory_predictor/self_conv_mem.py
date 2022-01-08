#!/usr/bin/env python3
from __future__ import division
from typing import Any, NamedTuple
import heapq
import numpy as np

from numpy.lib.function_base import place


# convolutional layer
class Conv_layer_param(NamedTuple):
    name: str
    in_h: int
    in_w: int
    in_c: int
    kernel_size: int
    padding: int
    stride: int
    out_h: int
    out_w: int
    out_c: int

class TensorInfo():
    def __init__(self, hi, wi, c):
        self.hi = hi
        self.wi = wi
        self.c = c
        self.lastchild = None
        self.start = 0
    

    def add_child(self, child):
        self.lastchild = child

    def get_lastchild(self):
        return self.lastchild

class MemoryAllocation():

    def __init__(self, layer_param):
        #print(layer_param)
        self.in_h = layer_param.in_h
        self.in_w = layer_param.in_w
        self.in_c = layer_param.in_c
        self.kernel_size = layer_param.kernel_size
        self.padding = layer_param.padding
        self.stride = layer_param.stride
        self.out_h = layer_param.out_h
        self.out_w = layer_param.out_w
        self.out_c = layer_param.out_c
        


    def get_lastchild(self, in_hi, in_wi):
        out_hi = max(0, min(int(np.floor((in_hi+self.padding)/self.stride)), self.out_h-1))
        out_wi = max(0, min(int(np.floor((in_wi+self.padding)/self.stride)), self.out_w-1))
        return out_hi, out_wi

    
    def actual_mem_size(self):
        # iterate through input tensors
        # find the last child in output FM according to depdency graph
        # place input tensor at max(curlast, pos of last child of out tensor)
        curend = 0
        for in_hi in range(self.in_h):
            for in_wi in range(self.in_w):
                child_hi, child_wi = self.get_lastchild(in_hi, in_wi)
                outmem_pos_lastchild = (child_hi * self.out_w + child_wi + 1) * self.out_c
                curend = max(curend, outmem_pos_lastchild)
                curend += self.in_c


        #print(curend, self.in_h * self.in_w * self.in_c, self.out_h * self.out_w * self.out_c)
        return curend


    def baseline_mem_size(self):
        return self.in_h * self.in_w * self.in_c + self.out_h * self.out_w * self.out_c
        

    def ideal_mem_size(self):
        return max(self.in_h * self.in_w * self.in_c, self.out_h * self.out_w * self.out_c)
