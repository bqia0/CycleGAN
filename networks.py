
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils
import os
from PIL import Image


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
        self.G_AB = ResnetGenerator(input_channels=3, output_channels=3) # A-> B
        self.G_BA = ResnetGenerator(input_channels=3, output_channels=3) # B-> A

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

        # Transforms
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2c5f2b14a577753b6ce40716e42dc28b21ed775a/data/base_dataset.py#L81
        # and from default base options
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
        self.train_transforms = transforms.Compose([
            transforms.Resize(286, Image.BICUBIC),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


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


    def get_dataloaders(self):
        data_a = datasets.ImageFolder('./datasets/summer2winter_yosemite/trainA', transform=self.train_transforms)
        data_b = datasets.ImageFolder('./datasets/summer2winter_yosemite/trainB', transform=self.train_transforms)
        loader_a = torch.utils.data.DataLoader(data_a,
                                          batch_size=1, 
                                          shuffle=True, 
                                          num_workers=1)
        loader_b = torch.utils.data.DataLoader(data_b,
                                          batch_size=1, 
                                          shuffle=True, 
                                          num_workers=1)
        
        return loader_a, loader_b

    def train(self):
        # Obtain dataloaders
        loader_a, loader_b = self.get_dataloaders()

        # Generated image pools
        imagepool_a = utils.ImagePool()
        imagepool_b = utils.ImagePool()

        # TODO: use arguments for hyperparameters
        lambda_coef = 10
        lambda_idt = 0.5

        step = 0

        self.load_checkpoint('./ckpts/checkpoint.ckpt')

        for epoch in range(self.curr_epoch, 200):

            for a_real, b_real in zip(loader_a, loader_b):
                # Send data to (ideally) GPU
                a_real = a_real.to(self.device)
                b_real = b_real.to(self.device)

                # batch size
                batch_size = a_real.shape[0]
                positive_labels = torch.ones(batch_size).to(self.device)
                negative_labels = torch.zeros(batch_size).to(self.device)
                
                # Generator forward passes
                a_fake = self.G_BA(b_real)
                b_fake = self.G_AB(a_real)

                a_reconstruct = self.G_BA(b_fake)
                b_reconstruct = self.G_AB(a_fake)

                a_identity = self.G_BA(a_real)
                b_identity = self.G_AB(b_real)

                # Identity Loss
                a_idt_loss = self.L1(a_identity, a_real) * lambda_coef * lambda_idt 
                b_idt_loss = self.L1(b_identity, b_real) * lambda_coef * lambda_idt 

                # GAN Loss
                a_fake_dis = self.D_A(a_fake)
                b_fake_dis = self.D_B(b_fake)

                a_gan_loss = self.MSE(a_fake_dis, positive_labels)
                b_gan_loss = self.MSE(b_fake_dis, positive_labels)

                # Cycle Loss
                a_cycle_loss = self.L1(a_reconstruct, a_real) * lambda_coef
                b_cycle_loss = self.L1(b_reconstruct, b_real) * lambda_coef

                # Total Loss
                total_gan_loss = a_idt_loss + b_idt_loss + a_fake_dis + a_gan_loss + b_gan_loss + a_cycle_loss + b_cycle_loss

                # Sample previously generated images for discriminator forward pass
                a_fake = imagepool_a(a_fake) # a_fake first dim might be batch entry
                b_fake = imagepool_b(b_fake)

                # Discriminator forward pass
                a_real_dis = self.D_A(a_real)
                a_fake_dis = self.D_B(a_fake)
                b_real_dis = self.D_B(b_real)
                b_fake_dis = self.D_B(b_fake)

                # Discriminator Losses
                a_dis_real_loss = self.MSE(a_real_dis, positive_labels)
                a_dis_fake_loss = self.MSE(a_fake_dis, negative_labels)
                b_dis_real_loss = self.MSE(b_real_dis, positive_labels)
                b_dis_fake_loss = self.MSE(b_fake_dis, negative_labels)

                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5

                # Step
                self.gen_optimizer.zero_grad()
                total_gan_loss.backwards()
                self.gen_optimizer.step()

                self.dis_optimizer.zero_grad()
                a_dis_loss.step()
                b_dis_loss.step()
                self.dis_optimizer.step()

                self.gen_scheduler.step()
                self.dis_scheduler.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
                                                (epoch, step + 1, min(len(loader_a), len(loader_b)),
                                                                total_gan_loss,a_dis_loss+b_dis_loss))

                step += 1
            self.save_checkpoint(epoch, './ckpts/checkpoint.ckpt')
            step = 0


        





