from torchvision import models
from torch.nn import Module, Sequential, ReLU, Conv2d, BatchNorm2d, ConvTranspose2d, DataParallel
import torch
from torch.nn.functional import interpolate


class UNET_VGG(Module):
    def __init__(self):
        super(UNET_VGG, self).__init__()
        self.vgg16_features = models.vgg16(pretrained=True).features
        """
            ### LAYER 1 ###
            features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.1 ReLU(inplace)
            features.2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.3 ReLU(inplace) -> (1, 64, 572, 572) !!!

            features.4 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            -> (, 64, 286, 286)
            
            ### LAYER 2 ###
            features.5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.6 ReLU(inplace)
            features.7 Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.8 ReLU(inplace) -> (, 128, 286, 286) !!!
            
            features.9 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            -> (, 128, 143, 143)

            ### LAYER 3 ###
            features.10 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.11 ReLU(inplace)
            features.12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.13 ReLU(inplace)
            features.14 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.15 ReLU(inplace) -> (, 256, 143, 143) !!!

            features.16 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            -> (, 256, 71, 71)

            ### LAYER 4 ###
            features.17 Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.18 ReLU(inplace)
            features.19 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.20 ReLU(inplace)
            features.21 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.22 ReLU(inplace) -> (, 512, 71, 71) !!!
            
            features.23 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            -> (, 512, 35, 35)

            ### LAYER 5 ###
            features.24 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.25 ReLU(inplace)
            features.26 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.27 ReLU(inplace)
            features.28 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            features.29 ReLU(inplace) -> (, 512, 35, 35) !!!

            >> maybe exlclude this
            features.30 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        """
        # Layer 1
        # (, 128, 286, 286)
        self.uplayer_1_conv1 = Conv2d(128, 64, 3, padding=(1, 1))
        self.uplayer_1_conv2 = Conv2d(64, 64, 3, padding=(1, 1))
        self.uplayer_1_conv3 = Conv2d(64, 1, 1)
        self.up_batch_norm_1_1 = BatchNorm2d(64)
        self.up_batch_norm_1_2 = BatchNorm2d(64)
        self.upconv_1 = ConvTranspose2d(128,
                                        64,
                                        2,
                                        padding=1,
                                        dilation=2,
                                        output_padding=0)
        # Layer 2
        # (, 128, 286, 286)
        self.uplayer_2_conv1 = Conv2d(256, 128, 3, padding=(1, 1))
        self.uplayer_2_conv2 = Conv2d(128, 128, 3, padding=(1, 1))
        self.up_batch_norm_2_1 = BatchNorm2d(128)
        self.up_batch_norm_2_2 = BatchNorm2d(128)
        self.upconv_2 = ConvTranspose2d(256,
                                        128,
                                        2,
                                        padding=1,
                                        dilation=2,
                                        output_padding=0)
        # Layer 3
        # (, 256, 143, 143)
        self.uplayer_3_conv1 = Conv2d(512, 256, 3, padding=(1, 1))
        self.uplayer_3_conv2 = Conv2d(256, 256, 3, padding=(1, 1))
        self.up_batch_norm_3_1 = BatchNorm2d(256)
        self.up_batch_norm_3_2 = BatchNorm2d(256)
        self.upconv_3 = ConvTranspose2d(512,
                                        256,
                                        2,
                                        padding=1,
                                        dilation=2,
                                        output_padding=1)
        # Layer 4
        # (, 512, 71, 71)
        self.uplayer_4_conv1 = Conv2d(1024, 512, 3, padding=(1, 1))
        self.uplayer_4_conv2 = Conv2d(512, 512, 3, padding=(1, 1))
        self.up_batch_norm_4_1 = BatchNorm2d(512)
        self.up_batch_norm_4_2 = BatchNorm2d(512)
        self.upconv_4 = ConvTranspose2d(512,
                                        512,
                                        2,
                                        padding=1,
                                        dilation=2,
                                        output_padding=1)

    def make_parallel(self):
        # Layer 1
        self.uplayer_1_conv1 = DataParallel(self.uplayer_1_conv1)
        self.uplayer_1_conv2 = DataParallel(self.uplayer_1_conv2)
        self.uplayer_1_conv3 = DataParallel(self.uplayer_1_conv3)
        self.upconv_1 = DataParallel(self.upconv_1)
        self.up_batch_norm_1_1 = DataParallel(self.up_batch_norm_1_1)
        self.up_batch_norm_1_2 = DataParallel(self.up_batch_norm_1_2)
        # Layer 2
        self.uplayer_2_conv1 = DataParallel(self.uplayer_2_conv1)
        self.uplayer_2_conv2 = DataParallel(self.uplayer_2_conv2)
        self.upconv_2 = DataParallel(self.upconv_2)
        self.up_batch_norm_2_1 = DataParallel(self.up_batch_norm_2_1)
        self.up_batch_norm_2_2 = DataParallel(self.up_batch_norm_2_2)
        # Layer 3
        self.uplayer_3_conv1 = DataParallel(self.uplayer_3_conv1)
        self.uplayer_3_conv2 = DataParallel(self.uplayer_3_conv2)
        self.upconv_3 = DataParallel(self.upconv_3)
        self.up_batch_norm_3_1 = DataParallel(self.up_batch_norm_3_1)
        self.up_batch_norm_3_2 = DataParallel(self.up_batch_norm_3_2)
        # Layer 4
        self.uplayer_4_conv1 = DataParallel(self.uplayer_4_conv1)
        self.uplayer_4_conv2 = DataParallel(self.uplayer_4_conv2)
        self.upconv_4 = DataParallel(self.upconv_4)
        self.up_batch_norm_4_1 = DataParallel(self.up_batch_norm_4_1)
        self.up_batch_norm_4_2 = DataParallel(self.up_batch_norm_4_2)

    def forward(self, input):
        output = self.vgg16_features[0](input)
        for layer in self.vgg16_features[1:4]:
            output = layer(output)
        layer1 = output
        output = self.vgg16_features[4](layer1)
        for layer in self.vgg16_features[5:9]:
            output = layer(output)
        layer2 = output
        output = self.vgg16_features[9](layer2)
        for layer in self.vgg16_features[10:16]:
            output = layer(output)
        layer3 = output
        output = self.vgg16_features[16](layer3)
        for layer in self.vgg16_features[17:23]:
            output = layer(output)
        layer4 = output
        output = self.vgg16_features[23](output)
        for layer in self.vgg16_features[24:30]:
            output = layer(output)
        layer5 = output

        # UPWARD PASS
        up_layer_4 = interpolate(layer5,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        up_layer_4 = self.upconv_4(up_layer_4)
        up_layer_4 = torch.cat((layer4, up_layer_4), dim=1)
        up_layer_4 = self.uplayer_4_conv1(up_layer_4)
        up_layer_4 = self.up_batch_norm_4_1(up_layer_4)
        up_layer_4 = ReLU(inplace=True)(up_layer_4)
        up_layer_4 = self.uplayer_4_conv2(up_layer_4)
        up_layer_4 = self.up_batch_norm_4_2(up_layer_4)
        up_layer_4 = ReLU(inplace=True)(up_layer_4)

        up_layer_3 = interpolate(up_layer_4,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        up_layer_3 = self.upconv_3(up_layer_3)
        up_layer_3 = torch.cat((layer3, up_layer_3), dim=1)
        up_layer_3 = self.uplayer_3_conv1(up_layer_3)
        up_layer_3 = self.up_batch_norm_3_1(up_layer_3)
        up_layer_3 = ReLU(inplace=True)(up_layer_3)
        up_layer_3 = self.uplayer_3_conv2(up_layer_3)
        up_layer_3 = self.up_batch_norm_3_2(up_layer_3)
        up_layer_3 = ReLU(inplace=True)(up_layer_3)

        up_layer_2 = interpolate(up_layer_3,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        up_layer_2 = self.upconv_2(up_layer_2)
        up_layer_2 = torch.cat((layer2, up_layer_2), dim=1)
        up_layer_2 = self.uplayer_2_conv1(up_layer_2)
        up_layer_2 = self.up_batch_norm_2_1(up_layer_2)
        up_layer_2 = ReLU(inplace=True)(up_layer_2)
        up_layer_2 = self.uplayer_2_conv2(up_layer_2)
        up_layer_2 = self.up_batch_norm_2_2(up_layer_2)
        up_layer_2 = ReLU(inplace=True)(up_layer_2)

        up_layer_1 = interpolate(up_layer_2,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        up_layer_1 = self.upconv_1(up_layer_1)
        up_layer_1 = torch.cat((layer1, up_layer_1), dim=1)
        up_layer_1 = self.uplayer_1_conv1(up_layer_1)
        up_layer_1 = self.up_batch_norm_1_1(up_layer_1)
        up_layer_1 = ReLU(inplace=True)(up_layer_1)
        up_layer_1 = self.uplayer_1_conv2(up_layer_1)
        up_layer_1 = self.up_batch_norm_1_2(up_layer_1)
        up_layer_1 = ReLU(inplace=True)(up_layer_1)
        up_layer_1 = self.uplayer_1_conv3(up_layer_1)
        output_layer = torch.sigmoid(up_layer_1)

        return output_layer
