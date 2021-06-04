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
    def __init__(self, hi, wi):
        self.hi = hi
        self.wi = wi
        self.ref_cnt = 0
        self.parents = []
        self.children = []
        self.start = 0
        self.size = 0
    
    def incr_ref(self):
        self.ref_cnt = self.ref_cnt + 1

    def decr_ref(self):
        self.ref_cnt = self.ref_cnt - 1

    def ref_cnt(self):
        return self.ref_cnt

    def assign_pos(self, start, size):
        self.start = start
        self.size = size

    def pos(self):
        return (self.start, self.size)

    def add_children(self, children):
        for child in children:
            self.children.append(child)

    def add_dependent(self, dependents):
        for dependent in dependents:
            self.parents.append(dependent)
            #dependent.add_children(self)
            dependent.incr_ref()

    # the computation for current tensor has finished
    # decrese the ref count for its dependecy
    # return list of new freed input tensors
    def remove_all_depedents(self):
        new_free_tensors = []
        for dependent in self.parents:
            #dependent.add_children(self)
            dependent.decr_ref()

            ##if ref count becomes 0, 
            # free the mem for this tensor, safe to rewrite
            if dependent.ref_cnt == 0:
                
                new_free_tensors.append(dependent)
        return new_free_tensors

    def print_dependents(self):
        for dependent in self.parents:
            print(dependent.hi, dependent.wi)

        


# val is a tuple
class Node(object):
    def __init__(self, val):
        self.val = val


    def __lt__(self, other):
        if type(self.val) is tuple:
            if self.val[0] < other.val[0]:
                return True
            return self.val[1] < other.val[1]
        elif type(self.val) is int:
            return self.val < other.val[1]

def find_input_window_tensors(hi, wi, kernel_size, h, w, tensors):
    res = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = hi + i
            y = wi + j
            if x >= 0 and y >= 0 and x < h and y < w:
                res.append(tensors[x][y])
    return res


class MemoryAllocation():

    def __init__(self, layer_param, prev_mem_layout=None):
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

        self.mem = prev_mem_layout
        self.intervals = []
        

        # create tensors for input FMs
        # in_h * in_w (each tensor has length of in_c, though not shown in self.tensors)
        self.in_tensors = [[TensorInfo(j,i) for i in range(self.in_w)] for j in range(self.in_h)]
        self.out_tensors = [[TensorInfo(j,i) for i in range(self.out_w)] for j in range(self.out_h)]

        if self.mem is None:
            #self.mem = MemoryLayout()
            self.mem = []
            self.init_mem_layerout()
            self.in_mem = []


        self.create_dependency()

        self.free_in_tensors = []
    

    # put the ouput FM sequentially in mem
    # the find how to padding input FM to
    # enable as much overwrite as possible
    def init_mem_layerout(self):

        for i in range(self.out_h):
            for j in range(self.out_w):
                tensor = self.out_tensors[i][j]
                #pos = self.assign_tensor((i,j), self.out_c)
                self.mem.extend([(i,j)]*self.out_c)
                tensor.assign_pos(len(self.mem) - self.out_c, self.out_c)


    # create dependency for input FMs and output FMs
    def create_dependency(self):
        #print(len(self.in_tensors[0]))
        for out_hi in range(self.out_h):
            for out_wi in range(self.out_w):
                # find the corresponding input window
                # add dependencies
                in_hi = out_hi * self.stride - self.padding
                in_wi = out_wi * self.stride - self.padding
                input_window_tensors = find_input_window_tensors(in_hi, in_wi, 
                    self.kernel_size, self.in_h, self.in_w, self.in_tensors)
                self.out_tensors[out_hi][out_wi].add_dependent(input_window_tensors)
                
    
    def place_inFM(self):
        for out_hi in range(self.out_h):
            for out_wi in range(self.out_w):
                #self.out_tensors[out_hi][out_wi].print_dependents()
                #print('remove dependent for', out_hi, out_wi)
                #self.out_tensors[out_hi][out_wi].print_dependents()
                new_free_tensors = self.out_tensors[out_hi][out_wi].remove_all_depedents()
                for tensor in new_free_tensors:
                    heapq.heappush(self.free_in_tensors, (tensor.hi, tensor.wi))
                
                self.assign_in_tensor((out_hi,out_wi))

        return


    def assign_in_tensor(self, tensor_idx):
        
        out_tensor = self.out_tensors[tensor_idx[0]][tensor_idx[1]]
        out_tensor_start, out_tensor_size = out_tensor.pos()
        cur_placed_at_least_end = out_tensor_start + out_tensor_size
        # if no free space to use yet
        # need padding for input tensors
        #print(type(self.free_in_tensors))
        
        if len(self.free_in_tensors) == 0:
            self.in_mem.extend([(-1,-1)] * (cur_placed_at_least_end-len(self.in_mem)))
        # put intput tensors in place
        else:
            # peek heap for smallest free input tensors
            #print(len(self.free_in_tensors))
            prev_idx = self.free_in_tensors[0]
            heapq.heappop(self.free_in_tensors)
            place_idx = [prev_idx]
            while len(self.free_in_tensors) > 0:
                cur_idx = self.free_in_tensors[0]
                # if next free input tensors are consecutive 
                # of the previous freed input tensors
                if (cur_idx[0] == prev_idx[0] and cur_idx[1] == prev_idx[1]+1) or (
                    cur_idx[0] == prev_idx[0]+1 and cur_idx[1] == 0 and prev_idx[1] == self.in_w-1):
                    heapq.heappop(self.free_in_tensors)
                    place_idx.append(cur_idx)
                    prev_idx = cur_idx

                else:
                    break
            
            # place found freed input tensors in mem
            # padding if needed
            if(len(place_idx)*self.in_c < cur_placed_at_least_end - len(self.in_mem)):
                self.in_mem.extend([[(-1,-1)] * (cur_placed_at_least_end -len(self.in_mem) - len(place_idx)*self.in_c)][0])
                regular_list = [[place_idx_i] * self.in_c for place_idx_i in place_idx]
                flat_list = [item for sublist in regular_list for item in sublist]
                self.in_mem.extend( flat_list)
                
            else:
                regular_list = [[place_idx_i] * self.in_c for place_idx_i in place_idx]
                flat_list = [item for sublist in regular_list for item in sublist]
                self.in_mem.extend(flat_list)

        #self.print_mem()

    def count_intervals(self):
        interval_start = 0
        interval_end = 0
        interval_cur = False
        for i,m in enumerate(self.in_mem):
            if m == (-1,-1):
                if interval_cur:
                    interval_end += 1
                else:
                    interval_start = i
                    interval_end = interval_start
                    interval_cur = True
            else:
                if interval_cur:
                    self.intervals.append([interval_start, interval_end])
                    interval_cur = False
                
        print(len(self.intervals))
        #print(self.intervals)

    def print_in_mem(self):
        print("size: ",len(self.in_mem))
        print(self.in_mem)

    def print_out_mem(self):
        print(self.mem)

    def save_mem(self, path):
        with open(path, "w") as output:
            output.write(str(self.in_mem))

    def in_mem_size(self):
        return len(self.in_mem)

    def actual_mem_size(self):
        return max(len(self.in_mem),self.out_h * self.out_w * self.out_c)

    def baseline_mem_size(self):
        return self.in_h * self.in_w * self.in_c + self.out_h * self.out_w * self.out_c

    def ideal_mem_size(self):
        return max(self.in_h * self.in_w * self.in_c, self.out_h * self.out_w * self.out_c)

