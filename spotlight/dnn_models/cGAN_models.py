import torch
import torch.nn as nn
import numpy as np
import torch

class parameter_learning(nn.Module):
    def __init__(self):
        super(parameter_learning, self).__init__()  


class generator(nn.Module):
    def __init__(self, noise_dim = 100, condition_dim=1447, layers=[150,300], output_dim = 5):
        super(generator, self).__init__()  

        self.z = noise_dim
        self.y = condition_dim
        self.output_dim = output_dim
        

        #List to store the dimensions of the layers
        self.layers = []
        self.softmax_list = []
        self.layerDims = layers.copy()
        self.layerDims.insert(0, self.z + self.y)
        
        
        for idx in range(len(self.layerDims)-1):
            self.layers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))
            self.layers.append(nn.BatchNorm1d(num_features=self.layerDims[idx+1]))
            # self.layers.append(nn.LeakyReLU(0.2,inplace=True))

        list_param = []
        
        for a in self.layers:
            list_param.extend(list(a.parameters()))

        self.fc_layers = nn.ParameterList(list_param)
        
        self.mult_heads =  nn.ModuleDict({})
        for b in range(self.output_dim):
            self.mult_heads['head_'+str(b)] =  nn.Sequential(nn.Linear(self.layerDims[-1], self.y))

        self.apply(self.init_weights)

    def forward(self, noise, condition,inference=False):

        # Returns multiple exits, one for each item.

        vector = torch.cat([noise, condition], dim=-1)

        for layers in self.layers:
            vector = layers(vector)
        if inference:
            outputs_tensors = []
            for output in self.mult_heads.values():
                out = output(vector)
                out = torch.tanh(out)
                _,indices = torch.max(out,1)
                outputs_tensors.append(indices)
            return outputs_tensors
        else:
            outputs_tensors = []
            for output in self.mult_heads.values():
                out = output(vector)
                out = torch.tanh(out)
                outputs_tensors.append(out)
            return tuple(outputs_tensors)

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
class discriminator(nn.Module):
    def __init__(self, condition_dim = 50 , layers = [300,150],input_dim=5, num_items=1447):
        super(discriminator, self).__init__()

        # Following the naming convention of https://arxiv.org/pdf/1411.1784.pdf
        
        self.slate_size = input_dim
        self.user_condition = condition_dim
        self.output_dim = 1
        self.num_items = num_items
        

        #List to store the dimensions of the layers
        self.layers = []
        self.layerDims = layers.copy()
        self.layerDims.insert(0, self.slate_size*self.num_items + self.user_condition)
        self.layerDims.append(self.output_dim)

        for idx in range(len(self.layerDims)-2):
            self.layers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))
            self.layers.append(nn.BatchNorm1d(num_features=self.layerDims[idx+1]))
            self.layers.append(nn.LeakyReLU(0.2))
        
        self.layers.append(nn.Linear(self.layerDims[-2], self.layerDims[-1]))

        list_param = []

        for a in self.layers:
            list_param.extend(list(a.parameters()))

        self.fc_layers = nn.ParameterList(list_param)

        self.apply(self.init_weights)

        

    def forward(self, batch_input, condition):
        # slate_batch = torchems
        vector = torch.cat([condition, batch_input], dim=-1).float()  # the concat latent vector
        for layers in self.layers:
            vector = layers(vector)
            vector = nn.functional.relu(vector) # Most probably, this has to change

        # return self.logistic(vector)
        return vector

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
