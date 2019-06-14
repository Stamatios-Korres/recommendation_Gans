import torch
import torch.nn as nn
import numpy as np
import torch

class parameter_learning(nn.Module):
    def __init__(self):
        super(parameter_learning, self).__init__()  


class generator(nn.Module):
    def __init__(self, noise_dim = 100, condition_dim=50, num_items = 1447, layers=[30],output_dim = 5):
        super(generator, self).__init__()  

        self.z = noise_dim
        self.y = condition_dim
        self.output_dim = output_dim
        self.num_items = num_items

        #List to store the dimensions of the layers
        self.layers = []
        self.softmax_list = []
        self.layerDims = layers.copy()
        self.layerDims.insert(0, self.z + self.y)
        
        
        for idx in range(len(self.layerDims)-1):
            self.layers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))
            self.layers.append(nn.LeakyReLU(0.2,inplace=True))

        list_param = []
        
        for a in self.layers:
            list_param.extend(list(a.parameters()))

        self.fc_layers = nn.ParameterList(list_param)
        
        self.mult_heads =  nn.ModuleDict({})
        for b in range(self.output_dim):
            self.mult_heads['head_'+str(b)] =  nn.Sequential(nn.Linear(self.layerDims[-1], self.num_items))

        self.apply(self.init_weights)

    def forward(self, noise, condition):

        # Returns multiple exits, one for each item.

        vector = torch.cat([noise, condition], dim=-1)  

        for layers in self.layers:
            vector = layers(vector)
        
        outputs_tensors = []
        for output in self.mult_heads.values():
            out = output(vector)
            outputs_tensors.append(out)
        return tuple(outputs_tensors)

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
class discriminator(nn.Module):
    def __init__(self, condition_dim = 50 , layers = [20],input_dim=5):
        super(discriminator, self).__init__()

        # Following the naming convention of https://arxiv.org/pdf/1411.1784.pdf
        
        self.x = input_dim
        self.y = condition_dim
        self.output_dim = 1

        #List to store the dimensions of the layers
        self.layers = []
        self.layerDims = layers.copy()
        self.layerDims.insert(0, self.y + self.x)
        self.layerDims.append(self.output_dim)

        for idx in range(len(self.layerDims)-1):
            self.layers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))
        list_param = []

        for a in self.layers:
            list_param.extend(list(a.parameters()))

        self.fc_layers = nn.ParameterList(list_param)

        self.apply(self.init_weights)

    def forward(self, Xu_input, condition):

        vector = torch.cat([condition, Xu_input], dim=-1)  # the concat latent vector

        for layers in self.layers[:-1]:
            vector = layers(vector)
            vector = nn.functional.relu(vector) # Most probably, this has to change

        return vector

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
