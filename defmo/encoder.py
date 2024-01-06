import torch.nn as nn
import torchvision.models

from torchvision.models.resnet import ResNet50_Weights


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # self.net = self.get_v1()
        # if args.encoder_architecture == 'v2':
        #     self.net = self.get_v2()
        self.net = self.get_v2()
        # self.net = self.get_v3()
        # elif args.encoder_architecture == 'v4':
        #     self.net = self.get_v4()
        # elif args.encoder_architecture == 'v5':
        #     self.net = self.get_v5()
        # else:
        #     raise RuntimeError('Unknown encoder architecture %s!', args.encoder_architecture)

    def get_v1(self):
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        modelc = nn.Sequential(*list(model.children())[:-2])
        pretrained_weights = modelc[0].weight
        modelc[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modelc[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        modelc[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
        return nn.Sequential(modelc, nn.PixelShuffle(2))

    def get_v2(self):
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        modelc1 = nn.Sequential(*list(model.children())[:3])
        modelc2 = nn.Sequential(*list(model.children())[4:8])
        pretrained_weights = modelc1[0].weight
        modelc1[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
        modelc = nn.Sequential(modelc1, modelc2)
        return modelc

    def get_v3(self):
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        modelc1 = nn.Sequential(*list(model.children())[:3])
        modelc2 = nn.Sequential(*list(model.children())[4:8])
        pretrained_weights = modelc1[0].weight
        modelc1[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
        modelc3 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        modelc = nn.Sequential(modelc1, modelc2, modelc3)
        return modelc

    def get_v4(self):
        model = torchvision.models.resnext50_32x4d(weights=ResNet50_Weights.IMAGENET1K_V1)
        modelc1 = nn.Sequential(*list(model.children())[:3])
        modelc2 = nn.Sequential(*list(model.children())[4:8])
        pretrained_weights = modelc1[0].weight
        modelc1[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
        modelc = nn.Sequential(modelc1, modelc2)
        return modelc

    def forward(self, inputs):
        return self.net(inputs)
