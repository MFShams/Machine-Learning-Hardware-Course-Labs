#######################################################################
# This code was written by:
# Mojtaba AlShams
#
# For questions and comments, Mojtaba can be reached on:
# University email: mojtaba.alshams@kaust.edu.sa
# Personal email: m.f.shams@hotmail.com
#######################################################################

import torch

if str(torch.randn(1).dtype) == 'torch.float32':
    dtypeSize = '4'
else:
    raise ValueError(f"invalid/unsupported dtype {str(torch.randn(1).dtype)}!")
model_configs = {}
def add_cnfig(name, layers, isTrained, inputSize, dtypeSize):
    model_configs[name] = {
        'name': name,
        'layers': layers,
        'pretrained': isTrained,
        'inputSize': inputSize,
        'dtypeSize': dtypeSize
    }
add_cnfig('ResNet152', '152', True, '224_224_3', dtypeSize)
add_cnfig('VGG19', '19', True, '224_224_3', dtypeSize)

def get_models_names():
    return list(model_configs.keys())

def get_cnfig(modelName):
    return model_configs[modelName]

def resnet152():
    isTrained = model_configs['ResNet152']['pretrained']
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=isTrained)

def vgg19():
    isTrained = model_configs['VGG19']['pretrained']
    return torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=isTrained)

