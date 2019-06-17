import torch, time, os, pickle
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import random
import logging
import tqdm
import copy
from torch.distributions import MultivariateNormal


from utils.storage_utils import save_statistics
from spotlight.dataset_manilupation import create_user_embedding
from spotlight.losses import (adaptive_hinge_loss, bpr_loss, hinge_loss, pointwise_loss)
from spotlight.factorization.representations import BilinearNet
from spotlight.evaluation import precision_recall_score_slates
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
from torch.autograd import gradcheck

logging.basicConfig(format='%(message)s',level=logging.INFO)


class CGAN(object):

    def __init__(self,  G=None,
                        D=None,
                        z_dim = 200,
                        n_iter = 15,
                        batch_size = 128,
                        l2 =0.0,
                        loss_fun = torch.nn.BCELoss(),
                        learning_rate=1e-4,
                        slate_size = 3,
                        G_optimizer_func=None,
                        D_optimizer_func=None,
                        experiment_name = "CGANs",
                        use_cuda=False,
                        alternate_k = 1,
                        random_state=None):

        self.exeriment_name = experiment_name
        self.experiment_folder = os.path.abspath("experiments_results/"+experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        self.starting_epoch = 0
        if not os.path.exists("experiments_results"):  # If experiment directory does not exist
            os.mkdir("experiments_results")

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

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
        if use_cuda:
            self.device = torch.device('cuda')
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            self.device = torch.device('cpu')
        print(torch.__version__)
        
        

    def _initialize(self):
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)

        self.G_Loss = nn.BCEWithLogitsLoss()
        self.D_Loss = nn.BCEWithLogitsLoss()
        for name, param in self.G.named_parameters():
            print(name)

        for name, param in self.D.named_parameters():
            print(name)

       
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
        one_hot =  torch.empty(0,slates.shape[1]*num_items).type(self.dtype)
        for z in slates:
            single_one_hot =  nn.functional.one_hot(z.to(torch.int64),num_classes = num_items).type(self.dtype)
            single_one_hot = single_one_hot.reshape(1,-1)
            one_hot = torch.cat((one_hot, single_one_hot), 0)
        return one_hot

    def fit(self,interactions,slates):

        self.num_users = interactions.shape[0]        
        self.num_items = interactions.shape[1]  
        self.train_user_embeddings = interactions
        self.train_slates = slates.astype(np.int)
        self._initialize()


        user_embedding_tensor = torch.from_numpy(self.train_user_embeddings).type(self.dtype)
        user_slate_tensor = torch.from_numpy(self.train_slates).type(self.dtype)
        logging.info('training start!!')
        
        total_losses = {"G_loss": [], "D_loss": [], "curr_epoch": []}


        
# In [34]: n.sample((2,))
        for epoch_num in range(self._n_iter):

            fake_score = 0 
            real_score = 0 

            g_train_epoch_loss = 0.0
            d_train_epoch_loss = 0.0
            
            current_epoch_losses = {"G_loss": [], "D_loss": []}

            #TODO: Check combination (batch_user,batch_slate)
            
            for minibatch_num, (batch_user,batch_slate) in enumerate(minibatch(user_embedding_tensor,user_slate_tensor,batch_size=self._batch_size)):
              
                self.D.train()
                self.G.train()
                
                # Use Soft and Noisy Labels 
                valid = (torch.ones(batch_user.shape[0], 1) * np.random.uniform(low=0.7, high=1.2, size=None)).type(self.dtype)
                fake = (torch.ones(batch_user.shape[0], 1) * np.random.uniform(low=0.0, high=0.3, size=None)).type(self.dtype)
                z = torch.rand(batch_user.shape[0],self.z_dim, device=self.device)
                

                # update D network
                self.D_optimizer.zero_grad()
                real_slates = self.one_hot_encoding(batch_slate,self.num_items)
                # Test discriminator on real images
                d_real_val = self.D(real_slates,batch_user,use_cuda = self._use_cuda)
                real_loss = self.D_Loss(d_real_val,valid)
                real_score += d_real_val.mean().item()
                # Test discriminator on fake images
                fake_slates = self.G(z,batch_user,use_cuda = self._use_cuda)
                fake_slates = torch.cat(fake_slates, dim=-1)
                d_fake_val = self.D(fake_slates.detach(),batch_user,use_cuda = self._use_cuda)
                fake_score += d_fake_val.mean().item()
                fake_loss = self.D_Loss(d_fake_val,fake)

                # Update discriminator parameter

                d_loss = fake_loss + real_loss
                d_train_epoch_loss += d_loss.item()
                d_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()
                
                fake_slates = self.G(z,batch_user,use_cuda = self._use_cuda)
                fake_slates = torch.cat(fake_slates, dim=-1)
                d_fake_val = self.D(fake_slates, batch_user,use_cuda = self._use_cuda)
                
                g_loss = self.G_Loss(d_fake_val, valid)
                g_train_epoch_loss+= g_loss.item()
                
                g_loss.backward()
                self.G_optimizer.step()

                current_epoch_losses["G_loss"].append(d_loss.item())         # add current iter loss to the train loss list
                current_epoch_losses["D_loss"].append(g_loss.item()) 
                

            total_losses['curr_epoch'].append(epoch_num)
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))
            

            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv', stats_dict=total_losses, 
                            current_epoch=epoch_num,continue_from_mode=True if (self.starting_epoch != 0 or epoch_num > 0) else False)
            g_train_epoch_loss /= minibatch_num
            d_train_epoch_loss /= minibatch_num

            logging.info("--------------- Epoch %d ---------------"%epoch_num)
            logging.info("Generator's loss: %f"%g_train_epoch_loss)
            logging.info("Discriminator's loss: %f"%d_train_epoch_loss)
            logging.info("Generator's score: %f"%fake_score)
            logging.info("Real score: %f"%real_score)
        try:
            state_dict_G = self.G.module.state_dict()
        except AttributeError:
            state_dict_G = self.G.state_dict()
        self.save_readable_model(self.experiment_saved_models, state_dict_G)

    def test(self,train,test,item_popularity,slate_size,precision_recall=True, map_recall= False):

        precision,recall = precision_recall_score_slates(self.G,test=test, train = train,
                                      k=self.slate_size, z_dim = self.z_dim,
                                      use_cuda=self._use_cuda)
        print(precision,recall)
   
    def save_readable_model(self, model_save_dir, state_dict):
        state ={'network': state_dict} # save network parameter and other variables.
        fname = os.path.join(model_save_dir, "generator")
        print('Saving state in ', fname)
        torch.save(state, f=fname)  # save state at prespecified filepath

    