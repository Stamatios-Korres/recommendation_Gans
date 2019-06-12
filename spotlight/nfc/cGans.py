import torch, time, os, pickle
import torch.nn as nn

import numpy as np
import torch
import torch.optim as optim
import random
import logging
import tqdm
import copy

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids

from spotlight.dataset_manilupation import create_user_embedding
from spotlight.losses import (adaptive_hinge_loss, bpr_loss, hinge_loss, pointwise_loss)

from spotlight.factorization.representations import BilinearNet
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
from spotlight.evaluation import rmse_score,precision_recall_score,evaluate_popItems,evaluate_random,hit_ratio

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format='%(message)s',level=logging.INFO)


class generator(nn.Module):
    def __init__(self, noise_dim = 100, input_dim=150,layers=[30,15],output_dim = 5):

        super(generator, self).__init__()

        self.z = noise_dim
        self.y = input_dim
        self.output_dim = output_dim

        #List to store the dimensions of the layers
        self.layers = []
        self.layerDims = layers.copy()
        self.layerDims.insert(0, self.z + self.x)
        self.layerDims.append(output_dim)

        for idx in range(len(self.layerDims)-1):
            self.layers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))
        list_param = []

        for a in self.layers:
            list_param.extend(list(a.parameters()))

        self.fc_layers = nn.ParameterList(list_param)

        self.apply(self.init_weights)

    def forward(self, noise, input):

        vector = torch.cat([noise, input], dim=-1)  # the concat latent vector

        for layers in self.layers[:-1]:
            vector = layers(vector)
            vector = nn.LeakyReLU(0.2,inplace=True)
        return vector

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
class disciminator(nn.Module):
    def __init__(self, condition_dim = 64 , layers = [20,15],input_dim=5):
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

    def forward(self, noise, input):

        vector = torch.cat([noise, input], dim=-1)  # the concat latent vector

        for layers in self.layers[:-1]:
            vector = layers(vector)
            vector = nn.functional.relu(vector) # Most probably, this has to change

        return vector

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class CGAN(object):

    def __init__(self,  G=None,
                        D=None,
                        z_dim = 64,
                        n_iter = 15,
                        batch_size = 128,
                        l2 =0.0,
                        loss_fun = torch.nn.MSELoss(), # torch.nn.BCELoss()
                        learning_rate=1e-4,
                        optimizer_func=None,
                        use_cuda=False,
                        sparse=False,
                        random_state=None):

        self._n_iter = n_iter
        self.G = G
        self.D = D
        self._learning_rate = learning_rate
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._optimizer_func = optimizer_func
        self._random_state = random_state or np.random.RandomState()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self._optimizer = None
        self._loss_func = loss_fun
        self.best_model = None
        self.best_validation = -1


        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()


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

    def fit(self,interactions,slates):

        self.num_users = interactions.shape[0]        
        self.num_items = interactions.shape[1]  
        self.user_embeddings = create_user_embedding(interactions)      
        self.slates = slates


        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        self.y_real_, self.y_fake_ = gpu(self.y_real_,self.use_cuda),  gpu(self.y_fake_,self.use_cuda)

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3).expand(self.batch_size, self.class_num, self.input_size, self.input_size)
                if self.gpu_mode:
                    x_, z_, y_vec_, y_fill_ = x_.cuda(), z_.cuda(), y_vec_.cuda(), y_fill_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_, y_fill_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_, y_fill_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_, y_fill_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1, (self.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = self.G(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))




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