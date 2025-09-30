import torch.nn.functional as F
from utils.model_config import CONFIGS_
import torch
import torchvision
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence
import collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_config import CONFIGS_
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MSPA(nn.Module):
    """高效多尺度病理感知注意力模块"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 多尺度空洞卷积金字塔
        self.pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 3, dilation=d, padding=d),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True)
            ) for d in [1, 2, 4]  # 细胞(1x1)、组织(2x2)、器官(4x4)
        ])

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * (in_channels // reduction), in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        pyramid_features = [branch(x) for branch in self.pyramid]
        # 特征融合
        fused = torch.cat(pyramid_features, dim=1)
        att_map = self.fusion(fused)
        return att_map


class ResNetMSPA(nn.Module):
    """带有MSPA注意力的ResNet模型"""

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        # 基础ResNet模型
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # # 将模型拆分为多个子模块，以便提取各层特征
        # self.conv1 = nn.Sequential(
        #     self.resnet.conv1,
        #     self.resnet.bn1,
        #     self.resnet.relu,
        #     self.resnet.maxpool
        # )
        # self.conv2_x = self.resnet.layer1
        # self.conv3_x = self.resnet.layer2
        # self.conv4_x = self.resnet.layer3
        # self.conv5_x = self.resnet.layer4
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self.encoder = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2, self.resnet.layer3,
            self.resnet.layer4,

        )
        # 分类器
        self.encoder1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                      nn.Flatten(), )

    def forward(self, x, return_features=False):
        if return_features:
            # features = []
            #
            # x = self.conv1(x)  # conv1
            # if return_features:
            #     features.append(x)
            #
            # x = self.conv2_x(x)  # conv2_x
            # if return_features:
            #     features.append(x)
            #
            # x = self.conv3_x(x)  # conv3_x
            # if return_features:
            #     features.append(x)
            #
            # x = self.conv4_x(x)  # conv4_x
            # if return_features:
            #     features.append(x)
            #
            # x = self.conv5_x(x)  # conv5_x
            # if return_features:
            #     features.append(x)
            #
            # x = self.avgpool(x)
            # x = torch.flatten(x, 1)
            # logits = self.classifier(x)
            # output = F.log_softmax(logits, dim=1)
            # return {
            #     'output': output,
            #     'features': features,
            #     'logit': logits
            # }
            return {}
        else:
            # x = self.conv1(x)
            # x = self.conv2_x(x)
            # features = x
            # x = self.conv3_x(x)
            # features = x
            # x = self.conv4_x(x)
            # # features = x
            # x = self.conv5_x(x)
            # features = x
            # x = self.avgpool(x)
            # # features = x
            # x = torch.flatten(x, 1)
            # # features = x
            x = self.encoder(x)
            features = x
            x = self.encoder1(x)
            logit = self.classifier(x)
            output = F.log_softmax(logit, dim=1)
            return {'output': output,
                    'features': features,
                    'logit': logit}


##### Neural Network model #####
#################################
class Net(nn.Module):
    def __init__(self, dataset='cifar', model='cnn'):
        super(Net, self).__init__()
        # define network layers
        print("Creating model for {}".format(dataset))
        self.dataset = dataset
        configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim=CONFIGS_[dataset]

        print('Network configs:', configs)
        self.named_layers, self.layers, self.layer_names =self.build_network(
            configs, input_channel, self.output_dim)
        self.n_parameters = len(list(self.parameters()))
        self.n_share_parameters = len(self.get_encoder())
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

        self.classifier = self.layers[self.layer_names.index('decode_fc2')]

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params


    def build_network(self, configs, input_channel, output_dim):
        layers = nn.ModuleList()
        named_layers = {}
        layer_names = []
        kernel_size, stride, padding = 3, 2, 1
        for i, x in enumerate(configs):
            if x == 'F':
                layer_name='flatten{}'.format(i)
                layer=nn.Flatten(1)
                layers+=[layer]
                layer_names+=[layer_name]
            elif x == 'M':
                pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                layer_name = 'pool{}'.format(i)
                layers += [pool_layer]
                layer_names += [layer_name]
            else:
                cnn_name = 'encode_cnn{}'.format(i)
                cnn_layer = nn.Conv2d(input_channel, x, stride=stride, kernel_size=kernel_size, padding=padding)
                named_layers[cnn_name] = [cnn_layer.weight, cnn_layer.bias]

                bn_name = 'encode_batchnorm{}'.format(i)
                bn_layer = nn.BatchNorm2d(x)
                named_layers[bn_name] = [bn_layer.weight, bn_layer.bias]

                relu_name = 'relu{}'.format(i)
                relu_layer = nn.ReLU(inplace=True)# no parameters to learn

                layers += [cnn_layer, bn_layer, relu_layer]
                layer_names += [cnn_name, bn_name, relu_name]
                input_channel = x

        # finally, classification layer
        fc_layer_name1 = 'encode_fc1'
        fc_layer1 = nn.Linear(self.hidden_dim, self.latent_dim)
        layers += [fc_layer1]
        layer_names += [fc_layer_name1]
        named_layers[fc_layer_name1] = [fc_layer1.weight, fc_layer1.bias]

        fc_layer_name = 'decode_fc2'
        fc_layer = nn.Linear(self.latent_dim, self.output_dim)
        layers += [fc_layer]
        layer_names += [fc_layer_name]
        named_layers[fc_layer_name] = [fc_layer.weight, fc_layer.bias]
        return named_layers, layers, layer_names


    def get_parameters_by_keyword(self, keyword='encode'):
        params=[]
        for name, layer in zip(self.layer_names, self.layers):
            if keyword in name:
                #layer = self.layers[name]
                params += [layer.weight, layer.bias]
        return params

    def get_encoder(self):
        return self.get_parameters_by_keyword("encode")

    def get_decoder(self):
        return self.get_parameters_by_keyword("decode")

    def get_shared_parameters(self, detach=False):
        return self.get_parameters_by_keyword("decode_fc2")

    def get_learnable_params(self):
        return self.get_encoder() + self.get_decoder()

    def forward(self, x, start_layer_idx = 0, logit=False):
        """
        :param x:
        :param logit: return logit vector before the last softmax layer
        :param start_layer_idx: if 0, conduct normal forward; otherwise, forward from the last few layers (see mapping function)
        :return:
        """
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        restults={}
        z = x
        for idx in range(start_layer_idx, len(self.layers)):
            layer_name = self.layer_names[idx]
            layer = self.layers[idx]
            z = layer(z)#.to(next(self.parameters()).device))
            if idx == 8:
                last_conv_output = z

        if self.output_dim > 1:
            restults['output'] = F.log_softmax(z, dim=1)
            restults['features']=last_conv_output

        else:
            restults['output'] = z
            restults['features']=last_conv_output

        if logit:
            restults['logit']=z
        return restults
    ###FedRCL
    # def forward(self, x, start_layer_idx=0, logit=False, return_features=False):
    #     results = {}
    #     z = x
    #     features = []
    #     for idx in range(start_layer_idx, len(self.layers)):
    #         layer_name = self.layer_names[idx]
    #         layer = self.layers[idx]
    #         z = layer(z)
    #         if 'encode_cnn' in layer_name:
    #             features.append(z)
    #     results['output'] = F.log_softmax(z, dim=1) if self.output_dim > 1 else z
    #     if return_features:
    #         results['features'] = features
    #     if logit:
    #         results['logit'] = z
    #     return results

    def mapping(self, z_input, start_layer_idx=-1, logit=True):

        z = z_input
        z=z.to(device)
        n_layers = len(self.layers)
        for layer_idx in range(n_layers + start_layer_idx, n_layers):
            layer = self.layers[layer_idx]
            z = layer(z)
        if self.output_dim > 1:
            out=F.log_softmax(z, dim=1)
        result = {'output': out}
        if logit:
            result['logit'] = z
        return result

