import torch, time, os, pickle
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import random
import logging
import tqdm
import copy




from spotlight.dataset_manilupation import create_user_embedding
from spotlight.losses import (adaptive_hinge_loss, bpr_loss, hinge_loss, pointwise_loss)
from spotlight.factorization.representations import BilinearNet
from spotlight.evaluation import rmse_score,precision_recall_score,evaluate_popItems,evaluate_random,hit_ratio
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
from torch.autograd import gradcheck

logging.basicConfig(format='%(message)s',level=logging.INFO)


class CGAN(object):

    def __init__(self,  G=None,
                        D=None,
                        z_dim = 100,
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

    def _initialize(self):
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
            self.G_Loss = nn.BCEWithLogitsLoss().cuda()
            self.D_Loss = nn.BCEWithLogitsLoss().cuda()
        else:
            self.G_Loss = nn.BCEWithLogitsLoss()
            self.D_Loss = nn.BCEWithLogitsLoss()

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
    
    def one_hot_encoding(self,slates,num_items):
        one_hot = torch.empty(0,slates.shape[1]*num_items)
        for i,z in enumerate(slates):
            single_one_hot =  nn.functional.one_hot(z.long(),num_classes = num_items)
            single_one_hot = single_one_hot.reshape(1,-1).float()
            one_hot = torch.cat((one_hot, single_one_hot), 0)
            
        return one_hot

    def fit(self,interactions,slates):

        self.num_users = interactions.shape[0]        
        self.num_items = interactions.shape[1]  
        self.train_user_embeddings = create_user_embedding(interactions)      
        self.train_slates = slates

        self._initialize()


        user_embedding_tensor = gpu(torch.from_numpy(self.train_user_embeddings.todense()), self._use_cuda)
        user_slate_tensor = gpu(torch.from_numpy(self.train_slates), self._use_cuda)

        logging.info('training start!!')
        
        for epoch_num in range(self._n_iter):
            fake_score = 0 
            real_score = 0 

            g_train_epoch_loss = 0.0
            d_train_epoch_loss = 0.0
            #TODO: Check combination (batch_user,batch_slate)

            for minibatch_num, (batch_user,batch_slate) in enumerate(minibatch(user_embedding_tensor,user_slate_tensor,batch_size=self._batch_size)):
              
                self.D.train()
                self.G.train()


                
                #one-sided label smoothing 
                valid = gpu(torch.ones(batch_user.shape[0], 1), self._use_cuda)
                fake = gpu(torch.zeros(batch_user.shape[0], 1), self._use_cuda)
                z = torch.from_numpy(np.random.normal(0, 1, (batch_user.shape[0], self.z_dim))).float()
                
                y = batch_user

                # update D network
                self.D_optimizer.zero_grad()
                real_slates = self.one_hot_encoding(batch_slate,self.num_items)

                # Test discriminator on real images
                d_real_val = self.D(real_slates.double(),y)
                real_loss = self.D_Loss(d_real_val,valid)
                real_score += d_real_val.mean().item()
                
                # Test discriminator on fake images
                fake_slates = torch.cat(self.G(z,y.float()),dim=-1)
                d_fake_val = self.D(fake_slates.detach().long(),y.long())
                fake_score += d_fake_val.mean().item()
                fake_loss = self.D_Loss(d_fake_val,fake)

                # Update discriminator parameter

                d_loss = fake_loss + real_loss
                d_train_epoch_loss += d_loss.item()
                d_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()
                
                d_fake_val = self.D(fake_slates.float(), y.float())
                
                g_loss = self.G_Loss(d_fake_val, fake)
                g_train_epoch_loss+= g_loss.item()
                g_loss.backward()
                self.G_optimizer.step()
            
            g_train_epoch_loss /= minibatch_num
            d_train_epoch_loss /= minibatch_num
            real_score /= minibatch_num
            fake_score /= minibatch_num
            # logging.info("Generator's loss: %f"%g_train_epoch_loss)
            logging.info("--------------- Epoch %d ---------------"%epoch_num)
            logging.info("Generator's score: %f"%fake_score)
            
            # logging.info("Discriminator's loss: %f"%d_train_epoch_loss)
            logging.info("Real score: %f"%real_score)

    def test(self,interactions,slates,item_popularity,slate_size,precision_recall=True, map_recall= False):
        self.test_user_embeddings = create_user_embedding(interactions)      
        self.test_slates = slates
        test_user_embedding_tensor = gpu(torch.from_numpy(self.train_user_embeddings.todense()), self._use_cuda)
        test_user_slate_tensor = gpu(torch.from_numpy(self.train_slates), self._use_cuda)

        
        for minibatch_num, (batch_user,batch_slate) in enumerate(minibatch(
                                                                test_user_embedding_tensor,
                                                                test_user_slate_tensor,
                                                                batch_size=self._batch_size)):
        
            z = torch.from_numpy(np.random.normal(0, 1, (batch_user.shape[0], self.z_dim))).float()
            fake_slates = self.G(z,batch_user.float(),inference = True)
            


    def create_slate(self,user_input):
        z = torch.from_numpy(np.random.normal(0, 1, (1, self.z_dim))).float()
        output = self.G(user_input,z)
        slate = []
        for item in output:
            _, indices = item.max(1)
            slate.append(indices.item())
        return slate


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


