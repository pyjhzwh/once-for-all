import sys
from numpy.lib.arraypad import pad
import torch
import numpy as np
from .utils import *
from .graph import *

# Python bindings generated by flatbuffers assume that they live on the top level,
# which they do not in here.
from .. import tflite
sys.modules["tflite"] = tflite


TFLITE_VERSION = 3
TFLITE_FILE_IDENTIFIER = "TFL" + str(TFLITE_VERSION)


def ComputePaddingWithOffset(stride, 
            dilation_rate, in_size, filter_size, out_size):
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    total_padding = ((out_size - 1) * stride + effective_filter_size - in_size)
    total_padding =  total_padding if (total_padding > 0) else 0
    offset = total_padding % 2
    return total_padding // 2#, offset      


def DoesEntryOverlapInTime(entry, first_used_at, last_used_at):
    if entry.tensor.first_used_at > last_used_at:
        return False
    if entry.tensor.last_used_at < first_used_at:
        return False
    return True

def NextSimultaneouslyActiveBuffer(first_entry, start, first_used_at, last_used_at):
    result = None
    candidate_next_entry = None
    if start is None:
        candidate_next_entry = first_entry
    else:
        if start.next_entry is None:
            return None
        candidate_next_entry = start.next_entry
        
    while(True):
        if (DoesEntryOverlapInTime(candidate_next_entry, first_used_at,
                                last_used_at)):
            result = candidate_next_entry
            break
        if (candidate_next_entry.next_entry is None):
            break
        candidate_next_entry = candidate_next_entry.next_entry
    
    return result



def CalForwardConv2DMemPaddingLen(op_params: , 
        weights , in_tensor: GTensor, out_tensor: GTensor) -> int:
    # calculate actual memory size
    curend = 0
    # n,h,w,c
    input_height = in_tensor.shape[1]
    input_width = in_tensor.shape[2]
    input_channel = in_tensor.shape[3]

    output_height = out_tensor.shape[1]
    output_width = out_tensor.shape[2]
    output_channel = out_tensor.shape[3]

    _, weights_arr = weights
    filter_height = weights_arr.shape[1]
    filter_width = weights_arr.shape[2]
    
    #padding = op_params.Padding() #Note: this is only padding type
    # PADDING_SAME:0; Padding_valid: 1
    dilation_height_factor = op_params.DilationHFactor()
    dilation_width_factor = op_params.DilationWFactor()
    stride_height = op_params.StrideH()
    stride_width = op_params.StrideW()
    padding_height = ComputePaddingWithOffset(stride_height, 
            dilation_height_factor, input_height, filter_height, output_height)
    padding_width = ComputePaddingWithOffset(stride_width, 
            dilation_width_factor, input_width, filter_width, output_width)
    # n,c,h,w
    for in_hi in range(input_height):
        for in_wi in range(input_width):
            # calculate the last child of in_hi, in_wi
            child_hi = max(0, min(output_height-1, 
                (int)(np.floor((in_hi+padding_height)/(stride_height))) ))
            child_wi = max(0, min(output_width-1, 
                (int)(np.floor((in_wi+padding_width)/(stride_width))) ))
            # need to +1, because output should not overwrite its dependent inputs
            outmem_pos_lastchild = (child_hi * output_width + child_wi + 1) * output_channel
            curend = max(curend, outmem_pos_lastchild)
            curend += input_channel
    
    return curend - in_tensor.size

def CalculatePaddingLen(
    op : GOperator, prior_tensor: GTensor,
    current_tensor: GTensor) -> int:

    if op.opcode == BuiltinOperator.CONV_2D:
        # https://github.com/jackwish/tflite/blob/master/tests/test_mobilenet.py
        # how to access conv2d options
        options = Conv2DOptions()
        options.Init(op.options.Bytes, op.options.Pos)
        # if not residual layer
        if prior_tensor.last_used_at == current_tensor.first_used_at:
            wanted_gap = CalForwardConv2DMemPaddingLen(options, op.weights, prior_tensor, current_tensor)
            return wanted_gap

    return prior_tensor.size

def CalCurrentOffset(
    prior_entry: ListEntry, prior_tensor: GTensor, 
    current_tensor: GTensor, space_prior_next: int) -> int:

    op = current_tensor.producer
    
    if op.opcode in OverlapOrInplaceOperator:
        # if prior buffer's consumer == current tensor's producer
        # the second == is to ensure it is the prior buffer will not be
        # used later, so we can safely overwrite it
        if op in prior_tensor.consumers and \
                prior_tensor.last_used_at == current_tensor.first_used_at:

            wanted_gap = CalculatePaddingLen(op, prior_tensor, current_tensor) 
            # If we could do forwarding computation
            if( space_prior_next >= wanted_gap):
                return prior_entry.offset - space_prior_next
            # Otherwise, reversed computation
            else:
                op.reverse = True
                padding = wanted_gap + prior_tensor.size - current_tensor.size
                return prior_entry.offset + padding
    
    return prior_entry.offset + prior_entry.size

def AlignSizeUp(size: int, alignment: int) -> int: 

    aligned_size = (((size + (alignment - 1)) // alignment) * alignment)
    return aligned_size


class MemoryPlanner:
    def __init__(self, model: TFLiteModel) -> None:
        self.model = model
        self.g = model.model_graph
        num_operators = len(self.g.operators)
        for t in self.g.tensors:
            t.compute_lifetime(num_operators)

        nonzero_tensors = [t for t in self.g.tensors if t.size !=0]
        self.tensors = nonzero_tensors
        self.first_entry = None
        self.need_to_calculate_offsets_= True
        

    def calculateOffsetIfNeeded(self):
        if self.need_to_calculate_offsets_ is False or len(self.tensors) == 0:
            return
        
        self.need_to_calculate_offsets_ = False

        # Sort tensors in ascending order of created_time, 
        # and then descending order of last_used time
        self.tensors.sort(key=lambda t:(t.first_used_at, -t.last_used_at))

        #for t in self.tensors:
        #    print(t.name, t.first_used_at, t.last_used_at)

        num_operators = len(self.g.operators)

        self.first_entry = ListEntry(self.tensors[0])
        self.tensors[0].offset = 0

        for i in range(1,len(self.tensors)):
            t = self.tensors[i]
            # unused tensor, do not need to allocate memory
            if(t.last_used_at == num_operators and t not in self.g.outputs):
                continue
            print(t.name)
            wanted_gap = t.size
            candidate_offset = 0
            prior_entry = None
            space_prior_next = 0
            while(True):
                # find the gap to place the current tensor;
                next_entry = NextSimultaneouslyActiveBuffer(self.first_entry,
                    prior_entry, t.first_used_at, t.last_used_at)

                # If we did not find a good gap in the previous steps
                if (prior_entry):
                    prior_entry_offset =  CalCurrentOffset(
                        prior_entry, prior_entry.tensor, t, space_prior_next)

                    aligned_prior_entry_offset = \
                        AlignSizeUp(prior_entry_offset, kBufferAlignment)
                    

                    if (aligned_prior_entry_offset > candidate_offset):
                        candidate_offset = aligned_prior_entry_offset
                    #print('candidate_offset',candidate_offset)

                if next_entry is None:
                    # We're at the end of the list, so we can always append the buffer
                    # here
                    break
                # Find out how much space there is between us and the next buffer.
                gap = next_entry.offset - candidate_offset
                wanted_gap = AlignSizeUp(t.size, kBufferAlignment)
                if (gap >= wanted_gap):
                    # This entry has a big enough gap between it and the next, so
                    # use it!
                    break
                # free space between prior entry and next_entry
                if prior_entry:
                    space_prior_next = next_entry.offset - (prior_entry.offset + prior_entry.size)
                else:
                    space_prior_next = next_entry.offset
                # The gap wasn't big enough, so move on to another candidate.
                prior_entry = next_entry

            # At this point, we've either found a gap (possibly at the end of the
            # list) and want to place the buffer there, or there are no other active
            # buffers in this time range and so we can put it at offset zero.
            # Record the buffer's offset in our plan.
            t.offset = candidate_offset
            # Add the newly-placed buffer to our offset-ordered list, so that
            # subsequent passes can fit in their buffers around it.
            new_entry = ListEntry(t)
            new_entry.offset = candidate_offset
            if (self.first_entry.offset > candidate_offset):
                # The new entry offset is smaller than the first entry offset =>
                # replace the first entry
                self.first_entry = new_entry
                self.first_entry.next_entry = self.first_entry
            else :
                current_entry = self.first_entry
                # Make sure that we insert the buffer at the correct place in the
                # buffer-offset-ordered list
                while True:
                    next_entry = current_entry.next_entry
                    if next_entry is None:
                        # We're at the end of the list, so just add the new entry here.
                        current_entry.next_entry = new_entry
                        break
                    
                    # not at the end of the list -> take a look at next entry
                    if (next_entry.offset > candidate_offset):
                        # We're at the right spot to do an insertion and retain the sorting
                        # order, so place the new entry here.
                        new_entry.next_entry = current_entry.next_entry
                        current_entry.next_entry = new_entry
                        break
                    current_entry = next_entry

    def printMemoryPlan(self):
        self.calculateOffsetIfNeeded()

        x = PrettyTable()
        x.field_names = ["i", "name", "size(B)", "offset", "Mem Usage(MB)", "first_used", "last_used"]
        #x.align["size(B)"] = "r"

        peak_mem_use =  0
        for i in range(len(self.tensors)):
            tensor = self.tensors[i]
            peak_mem_use = max(peak_mem_use, tensor.size + tensor.offset)
            x.add_row([i, tensor.name, tensor.size, tensor.offset, tensor.size + tensor.offset, tensor.first_used_at, tensor.last_used_at])


        print("Memory Plan")
        print(x)
        print(f"Current peak memory usage: {peak_mem_use:,} B")
        print()
