"""Taken & slightly modified from:
* https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import copy
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.last_relu:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.last_relu:
            out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        zero_init_residual=True,
        nf=16,
        last_relu=False,
        initial_kernel=3,
        shared_features=0,
        pretrained=None,
        **kwargs
    ):
        super(ResNet, self).__init__()
        assert shared_features <= 4
        
        self.shared_features = shared_features
        self.last_relu = last_relu
        self.inplanes = nf
        if initial_kernel == 3:
            self.conv1 = nn.Conv2d(3, nf, kernel_size=initial_kernel, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, nf, kernel_size=initial_kernel, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(nf)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 1 * nf, layers[0])
        self.layer2 = self._make_layer(block, 2 * nf, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * nf, layers[3], stride=2, last=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = 8 * nf * block.expansion
        print("Features dimension is {}.".format(self.out_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
        if pretrained is not None:
            print("Loading pretrained network")
            state_dict = model_zoo.load_url(pretrained)
            del state_dict["fc.weight"]
            del state_dict["fc.bias"]
            self.load_state_dict(state_dict)
            
        if self.shared_features == 0:
            self.shared = None
        else:
            if initial_kernel == 3:
                self.shared = nn.ModuleList([self.conv1])
            else:
                self.shared = nn.ModuleList([self.conv1, self.bn1, self.relu, self.maxpool])
                
            if self.shared_features > 1:
                self.shared.append(self.layer1)
            if self.shared_features > 2:
                self.shared.append(self.layer2)
            if self.shared_features > 3:
                self.shared.append(self.layer3)
                
        self.new = nn.ModuleList([self.layer4])
        if self.shared_features < 4:
            self.new.insert(0, self.layer3)
        if self.shared_features < 3:
            self.new.insert(0, self.layer2)
        if self.shared_features < 2:
            self.new.insert(0, self.layer1)
        if self.shared_features < 1:
            if initial_kernel != 3:
                self.new.insert(0, self.maxpool)
                self.new.insert(0, self.relu)
                self.new.insert(0, self.bn1)
            self.new.insert(0, self.conv1)
            
        self.old = None #nn.ModuleList([])
        
    def _make_layer(self, block, planes, blocks, stride=1, last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            if i == blocks - 1 and last:
                layers.append(block(self.inplanes, planes, last_relu=False))
            else:
                layers.append(block(self.inplanes, planes, last_relu=True))

        return nn.Sequential(*layers)

#     @property
#     def last_block(self):
#         return self.layer4

#     @property
#     def last_conv(self):
#         return self.layer4[-1].conv2

    def forward(self, x):
        if self.shared is not None:
            for layer in self.shared:
                x = layer(x)
        
        outputs = []
        if len(self.old) > 0:
            for layers in self.old:
                x_old = x
                for layer in layers:
                    x_old = layer(x_old)
                outputs.append(x_old)
        
        x_new = x
        for layer in self.new:
            x_new = layer(x_new)
        outputs.append(x_new)
        outputs = torch.cat(outputs, dim=1)
        
        raw_features = self.end_features(outputs)
        features = self.end_features(F.relu(outputs, inplace=False))

        return {
            "raw_features": raw_features,
            "features": features,
            "last_layer": outputs
        }

    def end_features(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def end_relu(self, x):
        if hasattr(self, "last_relu") and self.last_relu:
            return F.relu(x)
        return x

    def add_classes(self):
        if self.old is None:
            self.old = nn.ModuleList([])
        else:
            self.old.append(copy.deepcopy(self.new))        
        
    @property
    def new_weights(self):
        return self.new

    @property
    def shared_weights(self):
        return self.shared
    
    @property
    def old_weights(self):
        return self.old 
        
                            
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], pretrained=model_urls['resnet18'] if pretrained else None, **kwargs)
#     if pretrained:
#         print("Loading pretrained network")
#         state_dict = model_zoo.load_url(model_urls['resnet18'])
#         del state_dict["fc.weight"]
#         del state_dict["fc.bias"]
#         model.load_state_dict(state_dict)
    return model


def resnet32(**kwargs):
    model = ResNet(BasicBlock, [5, 4, 3, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], pretrained=model_urls['resnet50'] if pretrained else None, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print("Loading pretrained network")
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        model.load_state_dict(state_dict)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model