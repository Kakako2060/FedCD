
from utils.model_config import GENERATORCONFIGS
from utils.model_utils import get_dataset_name
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from FLAlgorithms.users.userFedCD import UserF
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import *
from FLAlgorithms.trainmodel.train_gan_fn import *
from FLAlgorithms.trainmodel.gan_models import *
from FLAlgorithms.trainmodel import gan_models
from torch.optim.lr_scheduler import StepLR
import copy
import matplotlib.pyplot as plt
MIN_SAMPLES_PER_LABEL = 1

#hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
#'mnist': (256, 32, 1, 10, 32),
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedFCD(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
        #print(clients)
        total_users = len(clients)
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()
        self.dataset = args.dataset
        self.dataset_name = get_dataset_name(self.dataset)
        # self.input_dim, self.h_dim, self.z_dim, self.input_channel, self.n_class, self.embedding_dim = GENERATORCONFIGS[
        #     self.dataset_name]
        self.ensemble_batch_size = args.batch_size
        self.ensemble_lr = args.ensemble_lr
        self.early_stop = 5
        self.train_data_loader, self.train_iter, self.available_labels = aggregate_user_data(data, args.dataset,
                                                                                             self.ensemble_batch_size)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=1e-5, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)
        self.total_train_samples = 0
        self.total_test_samples = 0
        self.train_losses = []
        self.test_losses = []


        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data(i, data, dataset=args.dataset, count_labels=True)
            #print("train_data['x'].shape:{} ".format(train_data['x'].shape))
            train_data, test_data = train_data, test_data
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            id, train, test = read_user_data(i, data, dataset=args.dataset)
            user = UserpFedFOR(
                args, id, model,
                train_data, test_data,
                self.available_labels, label_info,
                use_adam=self.use_adam)
            self.users.append(user)

        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print( )
        print("Finished creating FedFOR users.")

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def train(self, args):
        self.send_parameters(mode=self.mode)
        accuracy_list = []
        test_loss_list = []
        #### pretraining
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            # if glob_iter < self.early_stop:
            train_loss_avg = 0.0
            test_loss_avg = 0.0
            self.timestamp = time.time() # log user-training start time
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                train_loss=user.train(user_id,glob_iter,
                           early_stop=self.early_stop
                           )
                # for name, module in user.model.named_modules():
                #     print(name, module)
                curr_timestamp = time.time()  # log  user-training end time
                train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
                self.metrics['user_train_time'].append(train_time)
                train_loss_avg += train_loss
            # self.aggregate_parameters()
            self.metrics['train_loss'].append(train_loss_avg / len(self.selected_users))
            self.aggregate_parameters_CA()
            self.send_parameters(mode=self.mode)
            glob_acc, glob_loss = self.evaluate()
            accuracy_list.append(glob_acc)
            test_loss_list.append(glob_loss)
            self.test_losses.append(glob_loss)

        # self.plot_accuracy(accuracy_list)
        # self.plot_results(glob_acc)
        self.save_results(args)
        self.save_model()

    def plot_results(self,glob_acc):
        epochs = range(1, len(glob_acc) + 1)

        plt.figure(figsize=(20, 10))

        # Global Accuracy Plot
        plt.subplot(2, 2, 1)  # 2x2 grid, first subplot
        plt.plot(epochs, glob_acc, marker='o')
        plt.title('Global Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)

        # Training Loss Plot
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.train_losses, marker='o', label='Train Loss')
        plt.title('Optimization Error: Training Loss')
        plt.xlabel('Communication round T')
        plt.ylabel('Loss')
        plt.grid(True)

        # Testing Loss Plot
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.test_losses, marker='o', label='Test Loss')
        plt.title('Excess Risk: Test Loss')
        plt.xlabel('Communication round T')
        plt.ylabel('Loss')
        plt.grid(True)

        # Generalization Gap Plot
        plt.subplot(2, 2, 4)
        gen_gap = np.array(self.train_losses) - np.array(self.test_losses)
        plt.plot(epochs, gen_gap, marker='o', label='Generalization Gap')
        plt.title('Generalization Gap')
        plt.xlabel('Communication round T')
        plt.ylabel('Train Loss - Test Loss')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    def plot_accuracy(self, accuracy_list):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, marker='o')
        plt.title('Global Accuracy vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()


