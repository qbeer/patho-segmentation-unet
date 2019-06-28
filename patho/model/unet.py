from torch.nn import Module, Conv2d, ConvTranspose2d, MaxPool2d, DataParallel
from torch.nn.functional import pad, relu, interpolate
import torch


class UNET(Module):
    def __init__(self, input_shape=(572, 572, 3)):
        super(UNET, self).__init__()
        # Layer 1
        self.layer_1_conv1 = Conv2d(input_shape[-1], 64, 3)
        self.layer_1_conv2 = Conv2d(64, 64, 3)
        self.uplayer_1_conv1 = Conv2d(128, 64, 3)
        self.uplayer_1_conv2 = Conv2d(64, 64, 3)
        self.uplayer_1_conv3 = Conv2d(64, 1, 1)
        self.upconv_1 = ConvTranspose2d(
            128, 64, 2, padding=1, dilation=2, output_padding=0)
        # Layer 2
        self.layer_2_conv1 = Conv2d(64, 128, 3)
        self.layer_2_conv2 = Conv2d(128, 128, 3)
        self.uplayer_2_conv1 = Conv2d(256, 128, 3)
        self.uplayer_2_conv2 = Conv2d(128, 128, 3)
        self.upconv_2 = ConvTranspose2d(
            256, 128, 2, padding=1, dilation=2, output_padding=0)
        # Layer 3
        self.layer_3_conv1 = Conv2d(128, 256, 3)
        self.layer_3_conv2 = Conv2d(256, 256, 3)
        self.uplayer_3_conv1 = Conv2d(512, 256, 3)
        self.uplayer_3_conv2 = Conv2d(256, 256, 3)
        self.upconv_3 = ConvTranspose2d(
            512, 256, 2, padding=1, dilation=2, output_padding=0)
        # Layer 4
        self.layer_4_conv1 = Conv2d(256, 512, 3)
        self.layer_4_conv2 = Conv2d(512, 512, 3)
        self.uplayer_4_conv1 = Conv2d(1024, 512, 3)
        self.uplayer_4_conv2 = Conv2d(512, 512, 3)
        self.upconv_4 = ConvTranspose2d(
            1024, 512, 2, padding=1, dilation=2, output_padding=0)
        # Layer 5
        self.layer_5_conv1 = Conv2d(512, 1024, 3)
        self.layer_5_conv2 = Conv2d(1024, 1024, 3)

    def make_parallel(self):
        # Layer 1
        self.layer_1_conv1 = DataParallel(self.layer_1_conv1)
        self.layer_1_conv2 = DataParallel(self.layer_1_conv2)
        self.uplayer_1_conv1 = DataParallel(self.uplayer_1_conv1)
        self.uplayer_1_conv2 = DataParallel(self.uplayer_1_conv2)
        self.uplayer_1_conv3 = DataParallel(self.uplayer_1_conv3)
        self.upconv_1 = DataParallel(self.upconv_1)
        # Layer 2
        self.layer_2_conv1 = DataParallel(self.layer_2_conv1)
        self.layer_2_conv2 = DataParallel(self.layer_2_conv2)
        self.uplayer_2_conv1 = DataParallel(self.uplayer_2_conv1)
        self.uplayer_2_conv2 = DataParallel(self.uplayer_2_conv2)
        self.upconv_2 = DataParallel(self.upconv_2)
        # Layer 3
        self.layer_3_conv1 = DataParallel(self.layer_3_conv1)
        self.layer_3_conv2 = DataParallel(self.layer_3_conv2)
        self.uplayer_3_conv1 = DataParallel(self.uplayer_3_conv1)
        self.uplayer_3_conv2 = DataParallel(self.uplayer_3_conv1)
        self.upconv_3 = DataParallel(self.upconv_3)
        # Layer 4
        self.layer_4_conv1 = DataParallel(self.layer_4_conv1)
        self.layer_4_conv2 = DataParallel(self.layer_4_conv2)
        self.uplayer_4_conv1 = DataParallel(self.uplayer_4_conv1)
        self.uplayer_4_conv2 = DataParallel(self.uplayer_4_conv1)
        self.upconv_4 = DataParallel(self.upconv_4)
        # Layer 5
        self.layer_5_conv1 = DataParallel(self.layer_5_conv1)
        self.layer_5_conv2 = DataParallel(self.layer_5_conv1)

    def forward(self, x):
        # Down forward pass
        layer1 = self.layer_1_conv1(x)
        layer1 = relu(layer1)
        layer1 = self.layer_1_conv2(layer1)
        layer1 = relu(layer1)

        layer2 = MaxPool2d(2, stride=2)(layer1)
        layer2 = self.layer_2_conv1(layer2)
        layer2 = relu(layer2)
        layer2 = self.layer_2_conv2(layer2)
        layer2 = relu(layer2)

        layer3 = MaxPool2d(2, stride=2)(layer2)
        layer3 = self.layer_3_conv1(layer3)
        layer3 = relu(layer3)
        layer3 = self.layer_3_conv2(layer3)
        layer3 = relu(layer3)

        layer4 = MaxPool2d(2, stride=2)(layer3)
        layer4 = self.layer_4_conv1(layer4)
        layer4 = relu(layer4)
        layer4 = self.layer_4_conv2(layer4)
        layer4 = relu(layer4)

        layer5 = MaxPool2d(2, stride=2)(layer4)
        layer5 = self.layer_5_conv1(layer5)
        layer5 = relu(layer5)
        layer5 = self.layer_5_conv2(layer5)
        layer5 = relu(layer5)

        # Crops
        # TODO: needs te be modified to work for arbitrary input sizes
        # TODO: these are from the original UNET paper: https://arxiv.org/abs/1505.04597
        layer1_crop = pad(layer1, (-88, -88, -88, -88))
        layer2_crop = pad(layer2, (-40, -40, -40, -40))
        layer3_crop = pad(layer3, (-16, -16, -16, -16))
        layer4_crop = pad(layer4, (-4, -4, -4, -4))

        # Up forward pass
        up_layer_4 = interpolate(
            layer5, scale_factor=2, mode='bilinear', align_corners=True)
        up_layer_4 = self.upconv_4(up_layer_4)
        up_layer_4 = torch.cat((layer4_crop, up_layer_4), dim=1)
        up_layer_4 = self.uplayer_4_conv1(up_layer_4)
        up_layer_4 = relu(up_layer_4)
        up_layer_4 = self.uplayer_4_conv2(up_layer_4)
        up_layer_4 = relu(up_layer_4)

        up_layer_3 = interpolate(
            up_layer_4, scale_factor=2, mode='bilinear', align_corners=True)
        up_layer_3 = self.upconv_3(up_layer_3)
        up_layer_3 = torch.cat((layer3_crop, up_layer_3), dim=1)
        up_layer_3 = self.uplayer_3_conv1(up_layer_3)
        up_layer_3 = relu(up_layer_3)
        up_layer_3 = self.uplayer_3_conv2(up_layer_3)
        up_layer_3 = relu(up_layer_3)

        up_layer_2 = interpolate(
            up_layer_3, scale_factor=2, mode='bilinear', align_corners=True)
        up_layer_2 = self.upconv_2(up_layer_2)
        up_layer_2 = torch.cat((layer2_crop, up_layer_2), dim=1)
        up_layer_2 = self.uplayer_2_conv1(up_layer_2)
        up_layer_2 = relu(up_layer_2)
        up_layer_2 = self.uplayer_2_conv2(up_layer_2)
        up_layer_2 = relu(up_layer_2)

        up_layer_1 = interpolate(
            up_layer_2, scale_factor=2, mode='bilinear', align_corners=True)
        up_layer_1 = self.upconv_1(up_layer_1)
        up_layer_1 = torch.cat((layer1_crop, up_layer_1), dim=1)
        up_layer_1 = self.uplayer_1_conv1(up_layer_1)
        up_layer_1 = relu(up_layer_1)
        up_layer_1 = self.uplayer_1_conv2(up_layer_1)
        up_layer_1 = relu(up_layer_1)
        up_layer_1 = self.uplayer_1_conv3(up_layer_1)
        output_layer = torch.sigmoid(up_layer_1)

        return output_layer
