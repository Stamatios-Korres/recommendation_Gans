import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self,layers,num_users, num_items,output_dim = 1, embedding_dim=32):
        super(MLP, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = embedding_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.layers = []
        self.layerDims = layers.copy()
        self.layerDims.insert(0,2*embedding_dim)
        self.layerDims.append(output_dim)
        
        for idx in range(len(self.layerDims)-1):
            self.layers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))
        list_param = []
        for a in self.layers:
            list_param.extend(list(a.parameters()))

        self.fc_layers = nn.ParameterList(list_param)

        self.logistic = torch.nn.Sigmoid()
        self.apply(self.init_weights)

    def forward(self, user_indices, item_indices):

        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        

        for layers in self.layers[:-1]:
            vector = layers(vector)
            vector = nn.functional.relu(vector)
        logits = self.layers[-1](vector)
        rating = self.logistic(logits)
        return rating

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
