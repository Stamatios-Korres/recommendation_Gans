import torch
import torch.nn as nn
# from gmf import GMF
# from engine import Engine
# from utils import use_cuda, resume_checkpoint


class MLP(torch.nn.Module):
    def __init__(self,layers,num_users, num_items, embedding_dim=32):
        super(MLP, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = embedding_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.layers = []
        self.layerDims = layers.copy()
        self.layerDims.insert(0,2*embedding_dim)
        self.layerDims.append(1)

        for idx in range(len(self.layerDims)-1):
            self.layers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))

        list_param = []
        for a in self.layers:
            list_param.extend(list(a.parameters()))

        self.fc_layers = nn.ParameterList(list_param)

        self.affine_output = nn.Linear(in_features=self.layerDims[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

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

    def init_weight(self):
        pass

    # def load_pretrain_weights(self):
    #     """Loading weights from trained GMF model"""
    #     config = self.config
    #     gmf_model = GMF(config)
    #     if config['use_cuda'] is True:
    #         gmf_model.cuda()
    #     resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
    #     self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
    #     self.embedding_item.weight.data = gmf_model.embedding_item.weight.data

