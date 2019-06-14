import torch, time, os, pickle
import torch.nn as nn

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import random
import logging
import tqdm
import copy

from spotlight.dataset_manilupation import create_user_embedding
from spotlight.losses import (adaptive_hinge_loss, bpr_loss, hinge_loss, pointwise_loss)
from spotlight.factorization.representations import BilinearNet
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
from spotlight.evaluation import rmse_score,precision_recall_score,evaluate_popItems,evaluate_random,hit_ratio


logging.basicConfig(format='%(message)s',level=logging.INFO)


class CGAN(object):

    def __init__(self,  G=None,
                        D=None,
                        z_dim = 64,
                        n_iter = 15,
                        batch_size = 128,
                        l2 =0.0,
                        loss_fun = torch.nn.BCELoss(),
                        learning_rate=1e-4,
                        slate_size = 3,
                        G_optimizer_func=None,
                        D_optimizer_func=None,
                        use_cuda=False,
                        alternate_k = 1,
                        random_state=None):

        self._n_iter = n_iter
        self.G = G
        self.slate_size = slate_size
        self.alternate_k = 1
        self.D = D
        self._learning_rate = learning_rate
        self._l2 = l2
        self._use_cuda = use_cuda
        self.G_optimizer_func = G_optimizer_func
        self.D_optimizer_func = D_optimizer_func
        self._random_state = random_state or np.random.RandomState()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self._optimizer = None
        self._batch_size = batch_size
        self._loss_func = loss_fun
        
        self.G_use_dropout = False


    def one_hot_embedding(self,labels, num_classes):

        y = torch.eye(num_classes) 
        return y[labels]

    def _initialize(self):
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
            self.G_Loss = nn.BCELoss().cuda()
            self.D_Loss = nn.BCELoss().cuda()
        else:
            self.G_Loss = nn.BCELoss()
            self.D_Loss = nn.BCELoss()

        if self.G_optimizer_func is None:
            self.G_optimizer = optim.Adam(
                self.G.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
            self.D_optimizer = optim.Adam(
                self.D.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self.G_optimizer = self.G_optimizer_func(
                self.G.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
            self.D_optimizer = self.D_optimizer_func(
                self.D.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )

    def fit(self,interactions,slates):

        self.num_users = interactions.shape[0]        
        self.num_items = interactions.shape[1]  
        self.user_embeddings = create_user_embedding(interactions)      
        self.slates = slates

        self._initialize()

        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor    


        valid = Variable(FloatTensor(self._batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(self._batch_size, 1).fill_(0.0), requires_grad=False)

        user_embedding_tensor = gpu(torch.from_numpy(self.user_embeddings.todense()), self._use_cuda).float()
        user_slate_tensor = gpu(torch.from_numpy(self.slates), self._use_cuda).float()

        logging.info('training start!!')
        
        for epoch_num in range(self._n_iter):
            
            #TODO: Check combination (batch_user,batch_slate)

            for minibatch_num, (batch_user,batch_slate) in enumerate(minibatch(user_embedding_tensor,user_slate_tensor)):
                self.D.train()
                self.G.train()

                g_train_epoch_loss = 0.0
                d_train_epoch_loss = 0.0

                z = Variable(FloatTensor(np.random.normal(0, 1, (self._batch_size, self.z_dim))))
                y = batch_user

                # update D network
                self.D_optimizer.zero_grad()

                # Test discriminator on real images
                d_real_val = self.D(batch_slate,y)
                real_loss = self.D_Loss(d_real_val,valid)

                # Test discriminator on fake images
                gen_slate = self.G(z,y)
                d_fake_val = self.D(gen_slate,y)
                fake_loss = self.D_Loss(d_fake_val,fake)

                # Update discriminator parameter

                d_loss = fake_loss + real_loss
                d_train_epoch_loss+= d_loss.item()
                d_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                d_fake_val = self.D(gen_slate.detach(), y)
                g_loss = self.G_Loss(d_fake_val, valid)
                g_train_epoch_loss+= g_loss.item()
                g_loss.backward()
                self.G_optimizer.step()
            
            g_train_epoch_loss /= minibatch_num
            d_train_epoch_loss /= minibatch_num




# def save(self):
#     save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
#     torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

#     with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
#         pickle.dump(self.train_hist, f)

# def load(self):
#     save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

#     self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
#     self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))


# class generator(nn.Module):
#     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#     # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
#     def __init__(self, input_dim=100, output_dim=5, input_size=32, class_num=10):
#         super(generator, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.input_size = input_size
#         self.class_num = class_num

#         self.fc = nn.Sequential(
#             nn.Linear(self.input_dim + self.class_num, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
#             nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
#             nn.ReLU(),
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
#             nn.Tanh(),
#         )
#         utils.initialize_weights(self)

#     def forward(self, input, label):
#         x = torch.cat([input, label], 1)
#         x = self.fc(x)
#         x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
#         x = self.deconv(x)

#         return x
 
# class discriminator(nn.Module):

#     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#     # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
#     def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
#         super(discriminator, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.input_size = input_size
#         self.class_num = class_num

#         self.conv = nn.Sequential(
#             nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, self.output_dim),
#             nn.Sigmoid(),
#         )
#         utils.initialize_weights(self)

#     def forward(self, input, label):
#         '''
#             label: the condition given on top of the random noise
#             input: random choice generator from a normal distribution 
#         '''
#         x = torch.cat([input, label], 1)
#         x = self.conv(x)
#         x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
#         x = self.fc(x)

#         return x


  #     # fixed noise & condition
    #     self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
    #     for i in range(self.class_num):
    #         self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
    #         for j in range(1, self.class_num):
    #             self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

    #     temp = torch.zeros((self.class_num, 1))
    #     for i in range(self.class_num):
    #         temp[i, 0] = i

    #     temp_y = torch.zeros((self.sample_num, 1))
    #     for i in range(self.class_num):
    #         temp_y[i*self.class_num: (i+1)*self.class_num] = temp

    #     self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
    #     if self.gpu_mode:
    #         self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()