from ast import operator
import torch
from torch._C import Node
import torch.nn as nn
from .utils import *
from collections import OrderedDict

def get_tensor_shape(tensor):
    shape = tensor.type().sizes()
    # check if it has a shape:
    #shape = [d.dim_value for d in dim]
    return shape

def get_output_name(operator, i):
    return "Op_"+str(operator.id)+"_o"+str(i)

def get_type_element_size(t):
    sizes = {
        torch.int8:1,
        torch.uint8:1,
        torch.int16:2,
        torch.int32:4,
        torch.int64:8,
        torch.float32:4,
        torch.float64:8, 
    }
    return sizes[t]


class GOperator:
    def __init__(self, id=None, name=None, outputs=None, inputs=None, op=None, options=None, params=None):
        self.id = id
        self.name = name
        self.outputs = outputs
        self.inputs = inputs if inputs is not None else []
        self.op = op
        self.options = options
        self.params = params

    @property
    def non_empty_inputs(self):
        return [i for i in self.inputs if i is not None]

    #@property
    #def opcode_name(self):
    #    return OPCODE_NAMES[self.opcode]

    def __hash__(self):
        return hash(self.id)

class GTensor:
    def __init__(self, id=None, shape=None, name=None, is_constant=False, producer=None,
                 consumers=None, predecessors=None, type=None):
        self.id = id
        self.shape = shape
        self.name = name
        self.is_constant = is_constant
        self.producer = producer
        self.consumers = consumers if consumers is not None else []
        self.predecessors = predecessors
        self.type = type
        self.first_used_at = None
        self.last_used_at = None
        self.offset = 0
        self.reverse = False

    @property
    def size(self):
        return 0 if self.is_constant else torch.prod(self.shape) * get_type_element_size(self.type)

    def __hash__(self):
        return hash(self.id)

    def compute_lifetime(self, num_operators):
        if self.first_used_at and self.last_used_at:
            return
        self.first_used_at = self.producer.id if self.producer is not None else -1
        self.last_used_at = max(op.id for op in self.consumers) if self.consumers else num_operators

class Graph:
    def __init__(self, model=None, args=None) -> None:
        self.operators = {}
        self.tensors = {}
        self.edges = []
        self.model = model
        self.args = args

        if self.model is not None and self.args is not None:
            self._build_graph()
    
    # https://github.com/waleedka/hiddenlayer/blob/45243d51fd78cb6edc45cca50d29b04fb4b35511/hiddenlayer/pytorch_builder.py#L66
    def _build_graph(self, verbose=False):
        self.model.eval()
        # Run the Pytorch graph to get a trace and generate a graph from it
        with scope_name_workaround():
            trace, trace_outputs= torch.jit._get_trace_graph(self.model, self.args)
        torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.GOperatorExportTypes.ONNX)
        torch._C._jit_pass_inline(torch_graph)

        # create input tensor
        input_tensor = GTensor(id=0,shape=self.args.shape,name="input", producer=None)
    
        for nid, torch_node in enumerate(torch_graph.nodes()):
            
            # Op
            op = torch_node.kind()
            #print('op',op)
            # Parameters
            params = {k: torch_node[k] for k in torch_node.attributeNames()} 
            #print('params', params)
            # Inputs/outputs
            # TODO: inputs = [i.unique() for i in node.inputs()]
            inputs = torch_node.inputs() #[i.unique() for i in torch_node.inputs()]
            outputs = torch_node.outputs() #[o.unique() for o in torch_node.outputs()]
            #print(nid, torch_node)
            #print('inputs',inputs, 'outputs', outputs)
            # Get output shape
            #shape = get_shape(torch_node)
            # Add HL node
            operator = GOperator(id=nid, name=torch_node.scopeName(), op=op, 
                        outputs=outputs, inputs=None, params=params)
            #print(operator.name)
            self.add_operator(operator)
            
            # Assign outputs producer as operator
            for output_i, output in enumerate(outputs):
                #print(output.unique())
                output_id = output.unique()
                #print('output_id',output_id)
                if output_id not in self.tensors.keys():
                    #print(output.kind())
                    #print(get_tensor_shape(output))
                    output_tensor = GTensor(id=output_id, shape=get_tensor_shape(output), 
                        name=get_output_name(operator, output_i),
                        producer = operator, type=output.type().scalarType())
                    self.add_tensor(output_tensor)
                else:
                    raise ValueError("tensor with name{:} has been registered in tensors")
                    
            
            # Issue: Could not seperate featuremap and weights in nodes' inputs
            #for input in inputs:
            #    if len(input.uses()) == 0 :
            #        continue
            #    if input.type().kind() != 'ClassType':
            #        attributes = [attr for attr in dir(input) if not attr.startswith('__')]
            #        print('input', input.isCompleteGTensor(), input.offset(), input.requiresGrad(), input.type(), input.unique(),
            #            input.uses())
            
        
        # update tensors' consumers
        for torch_node in torch_graph.nodes():
            op_name = torch_node.scopeName()
            operator = self.operators[op_name]
            inputs = torch_node.inputs()
            for input in inputs:
                input_id = input.unique()
                #print('output_id',output_id)
                if input_id in self.tensors.keys():
                    target_tensor = self.tensors[input_id]
                    # append operator's inputs
                    operator.inputs.append(target_tensor)
                    # update target_tensor's consumer 
                    target_tensor.consumers.append(operator)
            # manually add input image tensor because we could not find a solution
            # to identify fmap and weights from node.inputs()
            if len(operator.inputs) == 0:
                operator.inputs.append(input_tensor)
                input_tensor.consumers.append(operator)
                    

        for k,v in self.operators.items():
            print(v.id, v.name, v.outputs, v.inputs)
        for k,v in self.tensors.items():
            print(k, v.id, v.name, v.shape, v.producer.name, v.consumers, v.type)
        

    
    def add_operator(self, operator):
        self.operators[operator.name] = operator

    def add_tensor(self, tensor):
        self.tensors[tensor.id] = tensor

    #def add_edge(self, node1, node2):
