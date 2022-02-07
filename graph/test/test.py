import torch
import torchvision.models
#import hiddenlayer as hl

from graph import Graph


model = torchvision.models.alexnet()

graph = Graph(model, torch.zeros([1,3,224,224]))
