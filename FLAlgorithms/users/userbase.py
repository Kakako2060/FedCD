import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from utils.model_utils import get_dataset_name
from utils.model_config import RUNCONFIGS
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from FLAlgorithms.trainmodel.gan_models import *

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


class User:
    """
    Base class for users in federated learning.
    """

    def __init__(self, args, id, model, train_data, test_data, use_adam=False):
        self.model = copy.deepcopy(model).to(device)  # Move model to device
        self.private_model = copy.deepcopy(model).to(device)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.K = args.K
        self.selected_users = []
        self.dataset = args.dataset
        # 新增模型类型参数 (关键修改点1)
        self.model_type = args.model.lower()

        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True, drop_last=True,
                                       )
        self.testloader = DataLoader(test_data, self.batch_size, drop_last=False
                                     )
        self.testloaderfull = DataLoader(test_data, self.test_samples )
        self.trainloaderfull = DataLoader(train_data, self.train_samples )
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        dataset_name = get_dataset_name(self.dataset)
        self.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.label_counts = {}

        self.local_model = copy.deepcopy(self.model).to(device)  # Move local model to device
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None

        self.init_loss_fn()
        if use_adam:
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            self.optimizer = pFedIBOptimizer(self.model.parameters(), lr=self.learning_rate)

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.label_counts = {}

    def init_loss_fn(self):
        self.loss = nn.NLLLoss().to(device)
        self.dist_loss = nn.MSELoss().to(device)
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean").to(device)
        self.ce_loss = nn.CrossEntropyLoss().to(device)

    #FedAWARE
    # def set_parameters(self, model, beta=1):
    #     for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(),
    #                                                  self.local_model.parameters()):
    #         if beta == 1:
    #             local_param.data = new_param.data.clone()
    #             old_param.data = new_param.data.clone()
    #         else:
    #             local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()
    #             old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()


    def set_parameters(self, model,beta=1):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()

    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()

    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    # only for pFedMAS
    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params, keyword='all'):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads


    # def test(self):
    #     self.model.eval()
    #     test_acc = 0
    #     loss = 0
    #     for x, y in self.testloaderfull:
    #         x = x.to(device)  # 确保输入数据在设备上
    #         y = y.to(device)  # 确保标签在设备上
    #         output = self.model(x)['output']
    #         loss += self.loss(output, y)
    #         test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #     return test_acc, loss, y.shape[0]
    def test(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        total_samples = 0
        # 关键修改点3：增加模型类型判断
        if self.model_type == 'lstm':
            for batch in self.testloaderfull:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
                y = batch['labels'].to(device)
                output = self.model(**inputs)['output']
                loss += self.loss(output, y).item()
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                total_samples += y.size(0)
        else:  # CNN等图像模型处理
            for x, y in self.testloaderfull:
                x, y = x.to(device), y.to(device)
                output = self.model(x)['output']
                loss += self.loss(output, y).item()*y.size(0)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                total_samples += y.size(0)
        return test_acc, loss, total_samples
    # def test_personalized_model(self):
    #     self.model.eval()
    #     test_acc = 0
    #     loss = 0
    #     self.update_parameters(self.personalized_model_bar)
    #     for x, y in self.testloaderfull:
    #         x = x.to(device)  # 将输入数据移动到设备上
    #         y = y.to(device)  # 将标签移动到设备上
    #         output = self.model(x)['output']
    #         loss += self.loss(output, y)
    #         test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #         # @loss += self.loss(output, y)
    #         # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
    #         # print(self.id + ", Test Loss:", loss)
    #     self.update_parameters(self.local_model)
    #     return test_acc, y.shape[0], loss
    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        total_samples = 0
        loss = 0
        self.update_parameters(self.personalized_model_bar)
        # 关键修改点4：个性化模型测试
        if self.model_type == 'lstm':
            for batch in self.testloaderfull:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
                y = batch['labels'].to(device)
                output = self.model(**inputs)['output']
                loss += self.loss(output, y).item()
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                total_samples += y.size(0)
        else:
            for x, y in self.testloaderfull:
                x, y = x.to(device), y.to(device)
                output = self.model(x)['output']
                loss += self.loss(output, y).item()
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                total_samples += y.size(0)
        self.update_parameters(self.local_model)
        return test_acc, total_samples, loss
    # def get_next_train_batch(self, count_labels=True):
    #     try:
    #         # Samples a new batch for personalizing
    #         (X, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (X, y) = next(self.iter_trainloader)
    #     result = {'X': X, 'y': y}
    #     if count_labels:
    #         unique_y, counts = torch.unique(y, return_counts=True)
    #         unique_y = unique_y.detach().numpy()
    #         counts = counts.detach().numpy()
    #         result['labels'] = unique_y
    #         result['counts'] = counts
    #     return result
    def get_next_train_batch(self, count_labels=True):
        try:
            batch = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            batch = next(self.iter_trainloader)

            # 关键修改点5：统一数据格式
        if self.model_type == 'lstm':
            x = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            y = batch['labels']
        else:
            x, y = batch[0], batch[1]

        result = {'X': x, 'y': y}
        if count_labels:
            unique_y, counts = torch.unique(y, return_counts=True)
            result['labels'] = unique_y.detach().cpu().numpy()
            result['counts'] = counts.detach().cpu().numpy()
        return result
    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))


