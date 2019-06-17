import torch
import torch.nn as nn
import numpy as np
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle

class parameter_learning(nn.Module):
    def __init__(self):
        super(parameter_learning, self).__init__()  


class generator(nn.Module):
    def __init__(self, noise_dim = 200, condition_dim=1447 , output_dim = 3):
        super(generator, self).__init__()  

        self.z = noise_dim
        self.y = condition_dim
        self.output_dim = output_dim
        
        #List to store the dimensions of the layers
        self.layers = nn.ModuleList()
        layers = [self.z + self.y, condition_dim]
        
        for idx in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[idx], layers[idx+1]))
            self.layers.append(nn.BatchNorm1d(num_features=layers[idx+1]))
            self.layers.append(nn.LeakyReLU(0.2,inplace=True))
        
        self.mult_heads =  nn.ModuleDict({})
        for b in range(self.output_dim):
            self.mult_heads['head_'+str(b)] =  nn.Sequential(nn.Linear(layers[-1], self.y))

        self.apply(self.init_weights)

    def forward(self, noise, condition,inference=False):

        # Returns multiple exits, one for each item.
        vector = torch.cat([noise, condition], dim=-1)
        for layers in self.layers:
            vector = layers(vector)
            outputs_tensors = []
        if inference:
            # Return the item in int format to suggest items to users
            for output in self.mult_heads.values():
                out = output(vector)
                out = torch.tanh(out)
                _,indices = torch.max(out,1)
                outputs_tensors.append(indices)
            slates = torch.empty([noise.shape[0],len(self.mult_heads)])
            for i,items in enumerate(zip(*tuple(outputs_tensors))):
                slates[i,:] = torch.stack(items)
            return slates
        else:
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
    def __init__(self, condition_dim = 100 ,  input_dim=3, num_items=1447):
        super(discriminator, self).__init__()

                
        self.slate_size = input_dim
        self.user_condition = condition_dim
        self.num_items = num_items
        

        #List to store the dimensions of the layers
        self.layers =  nn.ModuleList()
        layers = [self.slate_size*self.num_items + self.user_condition,num_items,1]

        for idx in range(len(layers)-2):
            self.layers.append(nn.Linear(layers[idx], layers[idx+1]))
            # self.layers.append(nn.BatchNorm1d(num_features=layers[idx+1]))
            self.layers.append(nn.Dropout(0.4))
            self.layers.append(nn.LeakyReLU(0.2))
        
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.apply(self.init_weights)

        

    def forward(self, batch_input, condition):

        vector = torch.cat([condition, batch_input], dim=-1).float() # the concat latent vector
        for layers in self.layers:
            vector = layers(vector)
        return vector

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

