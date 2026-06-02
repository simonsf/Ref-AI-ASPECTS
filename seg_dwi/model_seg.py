import torch
import torch.nn as nn
import torch.nn.functional as F

"""原论文中采用转置卷积进行上采样，这里给他改成原来的上采样方式"""
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            # nn.InstanceNorm2d(out_ch, affine=True),
            # nn.InstanceNorm2d(out_ch),
	    nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x
    

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,
                                                                                        nn.ConvTranspose2d) or isinstance(
                module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class encoder(nn.Module):
    def __init__(self, in_channels, initial_filter_size, kernel_size, do_instancenorm):
        super().__init__()
        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size,
                                       instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)

        self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)

        self.contr_4_1 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)

#         self.center = nn.Sequential(
#             nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.BatchNorm2d(initial_filter_size * 2 ** 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.BatchNorm2d(initial_filter_size * 2 ** 4),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        out = self.center(pool)
        return out, contr_4, contr_3, contr_2, contr_1
        
    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer
    
"""解码器1目前是解码的前三层"""
class decoder1(nn.Module):
    def __init__(self, initial_filter_size):
        super().__init__()
        self.upscale5 = up_conv(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
        
        self.upscale4 = up_conv(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)
        
        self.upscale3 = up_conv(initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_1 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2)
        

    def forward(self, x, contr_4, contr_3, contr_2):

        concat_weight = 1
        
        upscale = self.upscale5(x)
        crop = self.center_crop(contr_4, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)
        expand = self.expand_4_2(self.expand_4_1(concat))
        
        upscale = self.upscale4(expand)
        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)
        expand = self.expand_3_2(self.expand_3_1(concat))
        
        upscale = self.upscale3(expand)
        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)
        expand = self.expand_2_2(self.expand_2_1(concat))
        
        out = expand

        return out


    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        return layer    
    
    
"""解码器2目前是解码的最后一层"""    
class decoder2(nn.Module):
    def __init__(self, initial_filter_size, classes):
        super().__init__()
        
        self.upscale2 = up_conv(initial_filter_size * 2, initial_filter_size)
        self.expand_1_1 = self.expand(initial_filter_size * 2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        
        self.head = nn.Sequential(
                nn.Conv2d(initial_filter_size, classes, kernel_size=1,
                          stride=1, bias=False))

    def forward(self, x, contr_1):

        concat_weight = 1
        
        upscale = self.upscale2(x)
        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)
        expand = self.expand_1_2(self.expand_1_1(concat))
        
        out = self.head(expand)

        return out


    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        return layer

class Seg_d(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=1, do_instancenorm=True):
        super().__init__()
        
        self.encoder_dwi = encoder(in_channels, initial_filter_size, kernel_size, do_instancenorm)
        self.decoder1_dwi = decoder1(initial_filter_size)
        self.decoder2_dwi = decoder2(initial_filter_size, classes)

        
        self.active = torch.nn.Sigmoid()

    def forward(self, x):

        x_1, contr_4, contr_3, contr_2, contr_1 = self.encoder_dwi(x)
        decoder1_out = self.decoder1_dwi(x_1, contr_4, contr_3, contr_2)
        decoder2_out = self.decoder2_dwi(decoder1_out, contr_1)
        out = self.active(decoder2_out)
        return x_1, out
    
class Seg_f(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=1, do_instancenorm=True):
        super().__init__()
        
        self.encoder_flair = encoder(in_channels, initial_filter_size, kernel_size, do_instancenorm)
        self.decoder1_flair = decoder1(initial_filter_size)
        self.decoder2_flair = decoder2(initial_filter_size, classes)

        
        self.active = torch.nn.Sigmoid()

    def forward(self, x):

        x_1, contr_4, contr_3, contr_2, contr_1 = self.encoder_flair(x)
        decoder1_out = self.decoder1_flair(x_1, contr_4, contr_3, contr_2)
        decoder2_out = self.decoder2_flair(decoder1_out, contr_1)
        out = self.active(decoder2_out)
        return x_1, out
    
class Seg_t(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=1, do_instancenorm=True):
        super().__init__()
        
        self.encoder_tmax = encoder(in_channels, initial_filter_size, kernel_size, do_instancenorm)
        self.decoder1_tmax = decoder1(initial_filter_size)
        self.decoder2_tmax = decoder2(initial_filter_size, classes)

        
        self.active = torch.nn.Sigmoid()

    def forward(self, x):

        x_1, contr_4, contr_3, contr_2, contr_1 = self.encoder_tmax(x)
        decoder1_out = self.decoder1_tmax(x_1, contr_4, contr_3, contr_2)
        decoder2_out = self.decoder2_tmax(decoder1_out, contr_1)
        out = self.active(decoder2_out)
        return x_1, out
    
class PatchCL(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=1, do_instancenorm=True):
        super().__init__()
        
        self.encoder = encoder(in_channels, initial_filter_size, kernel_size, do_instancenorm)
        self.decoder1 = decoder1(initial_filter_size)
        
        self.head = nn.Sequential(
                    nn.Conv2d(initial_filter_size * 2, initial_filter_size * 2 , kernel_size=1, stride=2, padding=0, bias=True),
                    nn.BatchNorm2d(initial_filter_size * 2),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(initial_filter_size * 2, initial_filter_size * 2, kernel_size=1, stride=4, padding=0, bias=True),
            )
        
        
        self.apply(InitWeights_He(1e-2))
        
    def forward(self, x):

        x_1, contr_4, contr_3, contr_2, contr_1 = self.encoder(x)
        decoder1_out = self.decoder1(x_1, contr_4, contr_3, contr_2)
        head_out = self.head(decoder1_out)
        
        head_out = head_out.permute(0, 2, 3, 1)
        head_out = head_out.contiguous().view(-1,head_out.shape[3])

        return F.normalize(head_out, dim=1)   
    
class SliceCL(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=3, do_instancenorm=True):
        super().__init__()
        
        self.encoder = encoder(in_channels, initial_filter_size, kernel_size, do_instancenorm)

        self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4),
                nn.ReLU(inplace=True),
                nn.Linear(initial_filter_size * 2 ** 4, 128),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(initial_filter_size * 2, 128)
            )
#         #使用softmaxk函数
#         self.Softmax = nn.Softmax(dim=1)
        
        self.apply(InitWeights_He(1e-2))

    def forward(self, x):

        x_1, _, _, _, _ = self.encoder(x)
#         out = self.head(x_1)
        out = F.normalize(self.head(x_1), dim=1)
#         out = self.Softmax(out)
        
        return out