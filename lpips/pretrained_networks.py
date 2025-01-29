from collections import namedtuple
import torch
from torchvision import models as tv
from ultralytics import YOLO


class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple(
            "SqueezeOutputs",
            ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"],
        )
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)

        return out


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple(
            "AlexnetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"]
        )
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = tv.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = tv.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = tv.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = tv.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ["relu1", "conv2", "conv3", "conv4", "conv5"])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out


class yolov11m(torch.nn.Module):
    def __init__(
        self,
        requires_grad=False,
        pretrained=True,
        weighst_path="./lpips/weights/yolo11m.pt",
    ):
        super(yolov11m, self).__init__()

        if pretrained:
            yolo_model = YOLO(weighst_path)
            # Accessing the model from the ultralytics YOLO object
            self.model = yolo_model.model.model
        else:
            raise ValueError("Only pretrained model is available")

        # Dictionary to store the features
        self.features = {}

        # Hook for needed layers
        self.hooks = []
        self._register_hooks()

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output

            return hook

        # Registering hooks for the needed layers
        self.hooks.append(
            self.model[0].register_forward_hook(get_activation("conv1"))
        )  # Conv 64x160x160
        self.hooks.append(
            self.model[1].register_forward_hook(get_activation("conv2"))
        )  # Conv 128x80x80
        self.hooks.append(
            self.model[6].register_forward_hook(get_activation("c3k2_1"))
        )  # C3k2 512x20x20
        self.hooks.append(
            self.model[10].register_forward_hook(get_activation("c2psa"))
        )  # C2PSA 512x10x10
        self.hooks.append(
            self.model[16].register_forward_hook(get_activation("c3k2_2"))
        )  # C3k2 256x40x40
        self.hooks.append(
            self.model[22].register_forward_hook(get_activation("c3k2_3"))
        )  # C3k2 512x10x10

    def forward(self, x):
        # Clear the features dictionary
        self.features.clear()

        # Forward pass through the YOLO model
        self.model(x)

        outputs = namedtuple(
            "YOLOOutputs", ["conv1", "conv2", "c3k2_1", "c2psa", "c3k2_2", "c3k2_3"]
        )
        out = outputs(
            self.features["conv1"],
            self.features["conv2"],
            self.features["c3k2_1"],
            self.features["c2psa"],
            self.features["c3k2_2"],
            self.features["c3k2_3"],
        )

        return out

    def __del__(self):
        # Remove the hooks when the object is deleted
        for hook in self.hooks:
            hook.remove()
