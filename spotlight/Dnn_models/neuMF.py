import torch
import torch.nn as nn
from spotlight.factorization.representations import BilinearNet as GMF
from spotlight.nfc.mlp import MLP


class NeuMF(nn.Module):
    def __init__(self, mlp_layers,num_users, num_items,mf_embedding_dim=25, mlp_embedding_dim=32):
        super(NeuMF, self).__init__()
        # self.config = config
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_mf = mf_embedding_dim
        self.latent_dim_mlp = mlp_embedding_dim
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf )
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        # Initialize the neural network
        
        self.layers = []
        self.layerDims = mlp_layers.copy()
        self.layerDims.insert(0,2*mlp_embedding_dim)
        
        for idx in range(len(self.layerDims)-1):
            self.layers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))
        list_param = []
        for a in self.layers:
            list_param.extend(list(a.parameters()))

        self.fc_layers = nn.ParameterList(list_param)
        self.affine_output = nn.Linear(self.layerDims[-1] + mf_embedding_dim, 1)
        self.logistic = torch.nn.Sigmoid()

        self.apply(self.init_weights)

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)
        for layers in self.layers:
            mlp_vector = layers(mlp_vector)
            mlp_vector = nn.functional.relu(mlp_vector)
            # mlp_vector = torch.tanh(mlp_vector)
        
        vector = torch.cat([mlp_vector, mf_vector],dim=-1)


        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        return rating


    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    

   

    # def load_pretrain_weights(self):
    #     """Loading weights from trained MLP model & GMF model"""
    #     config = self.config
    #     config['latent_dim'] = config['latent_dim_mlp']
    #     mlp_model = MLP(config)
    #     if config['use_cuda'] is True:
    #         mlp_model.cuda()
    #     resume_checkpoint(mlp_model, model_dir=config['pretrain_mlp'], device_id=config['device_id'])

    #     self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
    #     self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
    #     for idx in range(len(self.fc_layers)):
    #         self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

    #     config['latent_dim'] = config['latent_dim_mf']
    #     gmf_model = GMF(config)
    #     if config['use_cuda'] is True:
    #         gmf_model.cuda()
    #     resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
    #     self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
    #     self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

    #     self.affine_output.weight.data = 0.5 * torch.cat([mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
    #     self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)
