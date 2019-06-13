import torch
import torch.nn as nn
import numpy as np
import torch



logging.basicConfig(format='%(message)s',level=logging.INFO)


class generator(nn.Module):
    def __init__(self, noise_dim = 100, input_dim=150,layers=[30,15],output_dim = 5):

        super(generator, self).__init__()

        self.z = noise_dim
        self.y = input_dim
        self.output_dim = output_dim

        #List to store the dimensions of the layers
        self.layers = []
        self.softmax_list = []
        self.layerDims = layers.copy()
        self.layerDims.insert(0, self.z + self.x)
        
        for idx in range(len(self.layerDims)-1):
            self.layers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))

        for idx in range(output_dim):
            self.softmax_list.append(nn.functional.log_softmax())

        list_param = []

        for a in self.layers:
            list_param.extend(list(a.parameters()))

        self.fc_layers = nn.ParameterList(list_param)

        self.apply(self.init_weights)

    def forward(self, noise, input):
        # Returns multiple exits, one for each item.

        vector = torch.cat([noise, input], dim=-1)  # the concat latent vector

        for layers in self.layers[:-1]:
            vector = layers(vector)
            vector = nn.LeakyReLU(0.2,inplace=True)

        softmax_results = []
        for softmax_list in self.softmax_list:
            logits = softmax_list[vector]
            # _, indices = torch.max(logits, 0)
            softmax_results.append(logits)
        return softmax_results

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
class disciminator(nn.Module):
    def __init__(self, condition_dim = 64 , layers = [20],input_dim=5):
        super(disciminator, self).__init__()

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
