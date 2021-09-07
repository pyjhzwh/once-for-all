#!/usr/bin/env python3
from __future__ import division
from typing import Any, NamedTuple
import heapq

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
        

        # create tensors for input FMs
        # in_h * in_w (each tensor has length of in_c, though not shown in self.tensors)
        self.in_tensors = [[TensorInfo(j,i,self.in_c) for i in range(self.in_w)] for j in range(self.in_h)]
        #self.out_tensors = [[LiteTensorInfo(j,i,self.out_c) for i in range(self.out_w)] for j in range(self.out_h)]

        self.create_dependency()

    
    def update_input_window_tensors_child(self, hi, wi, out_hi, out_wi, kernel_size, h, w, tensors):
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = hi + i
                y = wi + j
                if x >= 0 and y >= 0 and x < h and y < w:
                    tensors[x][y].add_child((out_hi,out_wi))
        return 

    # create dependency for input FMs and output FMs
    def create_dependency(self):
        #print(len(self.in_tensors[0]))
        for out_hi in range(self.out_h):
            for out_wi in range(self.out_w):
                # find the corresponding input window
                # add dependencies
                in_hi = out_hi * self.stride - self.padding
                in_wi = out_wi * self.stride - self.padding
                # only keep the last dependent out tensor 
                self.update_input_window_tensors_child(in_hi, in_wi, out_hi, out_wi,
                    self.kernel_size, self.in_h, self.in_w, self.in_tensors)

    
    def actual_mem_size(self):
        # iterate through input tensors
        # find the last child in output FM according to depdency graph
        # place input tensor at max(curlast, pos of last child of out tensor)
        curend = 0
        for in_hi in range(self.in_h):
            for in_wi in range(self.in_w):
                child_hi, child_wi = self.in_tensors[in_hi][in_wi].get_lastchild()
                outmem_pos_lastchild = (child_hi * self.out_w + child_wi) * self.out_c
                curend = max(curend, outmem_pos_lastchild)
                curend += self.in_c



        return curend


    def baseline_mem_size(self):
        return self.in_h * self.in_w * self.in_c + self.out_h * self.out_w * self.out_c
        

    def ideal_mem_size(self):
        return max(self.in_h * self.in_w * self.in_c, self.out_h * self.out_w * self.out_c)
