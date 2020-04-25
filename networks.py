
import torch
import torch.nn as nn
f
import utils
import os


class ResnetGenerator(nn.Module):
    """ Generator class utilizing resnets"""

    def __init__(self, input_channels, output_channels, ngf=64, normalization='instance', num_blocks=9):
        """
        Paramters:
            input_channels: number of channels in input image
            output_channels: number of channels in output image
            ngf: number of filters use in convolutions
            normalization: batch or instance
            num_blocks: number of resnet blocks. 9 for 256x256, 6 for lower
        """
        super(ResnetGenerator, self).__init__()
        model = []
        
        # Paper states to use instance norm, but we'll give the option of using batch
        if normalization == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif normalization == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('ResnetGenerator: Normaliztion [%s] is not implemented' % normalization)

        use_bias = False
        if normalization == 'instance':
            use_bias = True

        """ First Convolution Block"""
        # C7s1-64 block
        model += [nn.ReflectionPad2d(3), 
                  nn.Conv2d(input_channels, ngf, kernel_size=7, bias=use_bias), 
                  norm_layer(ngf), 
                  nn.ReLU(True)]

        """Downsampling"""
        # d128 block
        model += [nn.Conv2d(ngf, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(128),
                  nn.ReLU(True)]
        # d256 block
        model += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(256),
                  nn.ReLU(True)]

        """Resnet Blocks"""
        for _ in range(num_blocks):
            # R256 blocks
            model+=[ResnetBlock(256, norm_layer, use_bias)]
        
        """Upsampling"""
        # U128 block
        model += [nn.ConvTranspose2d(256, 128,
                                kernel_size=3, stride=2,
                                padding=1, output_padding=1,
                                bias=use_bias),
                  norm_layer(128),
                  nn.ReLU(True)]

        # U64 block
        model += [nn.ConvTranspose2d(128, 64,
                                kernel_size=3, stride=2,
                                padding=1, output_padding=1,
                                bias=use_bias),
                  norm_layer(64),
                  nn.ReLU(True)]
        
        """Final Convolution block"""
        # C7s1-3 block
        model += [nn.ReflectionPad2d(3), 
                  nn.Conv2d(64, output_channels, kernel_size=7, bias=use_bias)]

        # Last block uses tanh instead of ReLU
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__ (self, num_filters, norm_layer, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(num_filters, norm_layer, use_bias)
    
    def build_conv_block(self, num_filters, norm_layer, use_bias):
        """Image should come out the same size"""
        conv_block = []
        
        # Paper said reflection padding was used to reduce artifacts
        conv_block += [nn.ReflectionPad2d(1)]

        # Convolution with kernel size 3, and bias if InstanceNorm is used
        conv_block += [nn.Conv2d(num_filters, num_filters, kernel_size=3, bias=use_bias), 
                      norm_layer(num_filters), 
                      nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(num_filters, num_filters, kernel_size=3, bias=use_bias), 
                       norm_layer(num_filters)]

        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        return x + self.conv_block(x)

class Discriminator(nn.Module):
    """discriminator class trained alongisde generator"""

    def __init__(self, input_channels, ndf=64, n_layers=3, normalization='instance'):
        """
        Paramters:
            input_channels: number of channels in input image
            ndf: number of filters use in convolutions
            n_layers: number of conv layers
            normalization: batch or instance
        """
        super(Discriminator, self).__init__()
        model = []
        
        # Paper states to use instance norm, but we'll give the option of using batch
        if normalization == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif normalization == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('ResnetGenerator: Normaliztion [%s] is not implemented' % normalization)

        use_bias = False
        if normalization == 'instance':
            use_bias = True
        
        # C64 block
        model+=[nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1), 
                nn.LeakyReLU(0.2, True)]

        # C128, C256, C512 block
        for i in range(1, n_layers):
            mult_i = 2 ** (i-1)
            mult_o = 2 ** i
            model+=[nn.Conv2d(ndf * mult_i, ndf * mult_o, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ndf * mult_o), 
                    nn.LeakyReLU(0.2, True)]

        mult_i = 2 ** (n_layers-1)
        mult_o = 2 ** n_layers
        model+=[nn.Conv2d(ndf * mult_i, ndf * mult_o, kernel_size=4, stride=1, padding=1, bias=use_bias),
                norm_layer(ndf * mult_o), 
                nn.LeakyReLU(0.2, True)]
        
        # Final convolution
        model += [nn.Conv2d(ndf * mult_o, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class CycleGAN(object):
    """CycleGAN Class
    Contains both discriminators and generators
    and all optimizers/schedulers
    """
    def __init__(self, num_epochs, decay_epoch, initial_lr):
        # Generator Networks
        self.G_AB = ResnetGenerator(input_channels=3, output_channels=3)
        self.G_BA = ResnetGenerator(input_channels=3, output_channels=3)

        # Discriminator Networks
        self.D_A = Discriminator(input_channels=3)
        self.D_B = Discriminator(input_channels=3)

        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Losses
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Training items
        self.curr_epoch = 0

        self.gen_optimizer = torch.optim.Adam(list(self.G_AB.parameters()) + list(self.G_BA.parameters()), lr=initial_lr)
        self.dis_optimizer = torch.optim.Adam(list(self.D_A.parameters()) + list(self.D_B.parameters()), lr=initial_lr)

        self.gen_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_optimizer, utils.LambdaLR(num_epochs, decay_epoch).step)
        self.dis_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dis_optimizer, utils.LambdaLR(num_epochs, decay_epoch).step)

    def save_checkpoint(self, curr_epoch, ckpt_dir):
        state = {
            'epoch': curr_epoch,
            'G_AB': self.G_AB.state_dict(),
            'G_BA': self.G_BA.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'G_Optimizer': self.gen_optimizer.state_dict(),
            'D_Optimizer': self.dis_optimizer.state_dict(),
            # 'G_Scheduler': self.gen_scheduler.state_dict(),
            # 'D_Scheduler': self.dis_scheduler.state_dict()
        }
        torch.save(state, ckpt_dir)


    def load_checkpoint(self, ckpt_dir):
        if os.path.isfile(ckpt_dir):
            print("=> loading checkpoint '{}'".format(ckpt_dir))
            checkpoint = torch.load(ckpt_dir)
            self.curr_epoch = checkpoint['epoch']
            self.G_AB.load_state_dict(checkpoint['G_AB'])
            self.G_BA.load_state_dict(checkpoint['G_BA'])
            self.D_A.load_state_dict(checkpoint['D_A'])
            self.D_B.load_state_dict(checkpoint['D_B'])
            self.gen_optimizer.load_state_dict(checkpoint['G_Optimizer'])
            self.dis_optimizer.load_state_dict(checkpoint['D_Optimizer'])
            # self.gen_scheduler.load_state_dict(checkpoint['G_Scheduler'])
            # self.dis_scheduler.load_state_dict(checkpoint['D_Scheduler'])

            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_dir))







