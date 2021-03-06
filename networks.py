
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils
import os
from PIL import Image


class ResnetGenerator(nn.Module):
    """ Generator class utilizing resnets"""

    def __init__(self, input_channels, output_channels, ngf=64, normalization='instance', use_dropout=False, num_blocks=9):
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
            model+=[ResnetBlock(256, norm_layer, use_dropout, use_bias)]
        
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
    def __init__ (self, num_filters, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(num_filters, norm_layer, use_dropout, use_bias)
    
    def build_conv_block(self, num_filters, norm_layer, use_dropout, use_bias):
        """Image should come out the same size"""
        conv_block = []
        
        # Paper said reflection padding was used to reduce artifacts
        conv_block += [nn.ReflectionPad2d(1)]

        # Convolution with kernel size 3, and bias if InstanceNorm is used
        conv_block += [nn.Conv2d(num_filters, num_filters, kernel_size=3, bias=use_bias), 
                      norm_layer(num_filters), 
                      nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

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
    def __init__(self, args):
        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Generator Networks
        self.G_AB = ResnetGenerator(input_channels=3, output_channels=3, ngf=args.ngf, 
                                    normalization=args.norm, use_dropout=not args.no_dropout).to(self.device) # A-> B

        self.G_BA = ResnetGenerator(input_channels=3, output_channels=3, ngf=args.ngf, 
                                    normalization=args.norm, use_dropout=not args.no_dropout).to(self.device) # B-> A

        # Discriminator Networks
        if args.train:
            self.D_A = Discriminator(input_channels=3, ndf=args.ndf, normalization=args.norm).to(self.device)
            self.D_B = Discriminator(input_channels=3, ndf=args.ndf, normalization=args.norm).to(self.device)


            # Losses
            self.MSE = nn.MSELoss()
            self.L1 = nn.L1Loss()

            # Training items
            self.curr_epoch = 0

            self.gen_optimizer = torch.optim.Adam(list(self.G_AB.parameters()) + list(self.G_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
            self.dis_optimizer = torch.optim.Adam(list(self.D_A.parameters()) + list(self.D_B.parameters()), lr=args.lr, betas=(0.5, 0.999))

            self.gen_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_optimizer, lr_lambda=utils.LambdaLR(args.epochs, args.decay_epoch).step)
            self.dis_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dis_optimizer, lr_lambda=utils.LambdaLR(args.epochs, args.decay_epoch).step)

        # Transforms
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2c5f2b14a577753b6ce40716e42dc28b21ed775a/data/base_dataset.py#L81
        # and from default base options
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
        self.train_transforms = transforms.Compose([
            transforms.Resize(args.load_size, Image.BICUBIC),
            transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(args.crop_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def save_checkpoint(self, curr_epoch, args):
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
        file_dir = os.path.join(args.checkpoint_dir, args.dataset).replace("\\","/")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        torch.save(state, os.path.join(file_dir, 'checkpoint.ckpt').replace("\\","/"))


    def load_checkpoint(self,args):
        file_dir = os.path.join(args.checkpoint_dir, args.dataset, 'checkpoint.ckpt').replace("\\","/")
        if os.path.isfile(file_dir):
            print("=> loading checkpoint '{}'".format(file_dir))
            checkpoint = torch.load(file_dir)
            self.G_AB.load_state_dict(checkpoint['G_AB'])
            self.G_BA.load_state_dict(checkpoint['G_BA'])

            if args.train:
                self.curr_epoch = checkpoint['epoch']
                self.D_A.load_state_dict(checkpoint['D_A'])
                self.D_B.load_state_dict(checkpoint['D_B'])
                self.gen_optimizer.load_state_dict(checkpoint['G_Optimizer'])
                self.dis_optimizer.load_state_dict(checkpoint['D_Optimizer'])
            # self.gen_scheduler.load_state_dict(checkpoint['G_Scheduler'])
            # self.dis_scheduler.load_state_dict(checkpoint['D_Scheduler'])

            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(file_dir))


    def get_dataloader(self, args):

        if args.train:
            mode = 'train'
            transforms = self.train_transforms
        else:
            mode = 'test'
            transforms = self.test_transforms

        data = utils.CycleGANDataset('./datasets/'+args.dataset, transform=transforms, mode=mode)

        loader = torch.utils.data.DataLoader(data,
                                          batch_size=args.batch_size, 
                                          shuffle=True, 
                                          num_workers=1)
        
        return loader
    
    def test(self, args):
        loader = self.get_dataloader(args)
        self.load_checkpoint(args)

        self.G_BA.eval()
        self.G_AB.eval()

        i = 0

        results_dir = os.path.join(args.results_dir, args.dataset).replace("\\","/")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for a_real, b_real in loader:

            if i == args.test_samples:
                break

            a_real = a_real.to(self.device)
            b_real = b_real.to(self.device)

            with torch.no_grad():
                a_fake = self.G_BA(b_real)
                b_fake = self.G_AB(a_real)
                a_reconstruct = self.G_BA(b_fake)
                b_reconstruct = self.G_AB(a_fake)
            i+=1

            output_image = (torch.cat([a_real, b_fake, a_reconstruct, b_real, a_fake, b_reconstruct], dim=0).data + 1)/ 2.0 # why add 1 then devide?
            torchvision.utils.save_image(output_image, os.path.join(results_dir, 'test_{}.jpg'.format(i)).replace("\\","/"), nrow=3)


    def train(self, args):
        # Obtain dataloaders
        loader = self.get_dataloader(args)

        # Generated image pools
        imagepool_a = utils.ImagePool()
        imagepool_b = utils.ImagePool()

        lambda_coef = args.lamda
        lambda_idt = args.idt_coef

        # Initialize Weights
        utils.init_weights(self.G_BA)
        utils.init_weights(self.G_AB)
        utils.init_weights(self.D_A)
        utils.init_weights(self.D_B)

        step = 0

        self.load_checkpoint(args)

        # Terrible hack
        self.gen_scheduler.last_epoch = self.curr_epoch - 1
        self.dis_scheduler.last_epoch = self.curr_epoch - 1
        
        self.G_BA.train()
        self.G_AB.train()

        for epoch in range(self.curr_epoch, args.epochs):

            for a_real, b_real in loader:
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

                positive_labels = torch.ones_like(a_fake_dis)

                a_gan_loss = self.MSE(a_fake_dis, positive_labels)
                b_gan_loss = self.MSE(b_fake_dis, positive_labels)

                # Cycle Loss
                a_cycle_loss = self.L1(a_reconstruct, a_real) * lambda_coef
                b_cycle_loss = self.L1(b_reconstruct, b_real) * lambda_coef

                # Total Loss
                total_gan_loss = a_idt_loss + b_idt_loss + a_gan_loss + b_gan_loss + a_cycle_loss + b_cycle_loss

                # Sample previously generated images for discriminator forward pass
                a_fake = torch.Tensor(imagepool_a(a_fake.detach().cpu().clone().numpy())) # a_fake first dim might be batch entry
                b_fake = torch.Tensor(imagepool_b(b_fake.detach().cpu().clone().numpy()))

                a_fake = a_fake.to(self.device)
                b_fake = b_fake.to(self.device)

                # Discriminator forward pass
                a_real_dis = self.D_A(a_real)
                a_fake_dis = self.D_B(a_fake)
                b_real_dis = self.D_B(b_real)
                b_fake_dis = self.D_B(b_fake)

                # Discriminator Losses
                positive_labels = torch.ones_like(a_fake_dis)
                negative_labels = torch.zeros_like(a_fake_dis)

                a_dis_real_loss = self.MSE(a_real_dis, positive_labels)
                a_dis_fake_loss = self.MSE(a_fake_dis, negative_labels)
                b_dis_real_loss = self.MSE(b_real_dis, positive_labels)
                b_dis_fake_loss = self.MSE(b_fake_dis, negative_labels)

                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5

                # Step
                self.gen_optimizer.zero_grad()
                total_gan_loss.backward()
                self.gen_optimizer.step()

                self.dis_optimizer.zero_grad()
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.dis_optimizer.step()

                for group in self.dis_optimizer.param_groups:
                    for p in group['params']:
                        state = self.dis_optimizer.state[p]
                        if state['step'] >= 962:
                            state['step'] = 962

                for group in self.gen_optimizer.param_groups:
                    for p in group['params']:
                        state = self.gen_optimizer.state[p]
                        if state['step'] >= 962:
                            state['step'] = 962

                if (step + 1) % 5 == 0:
                    print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
                                                    (epoch, step + 1, len(loader),
                                                                    total_gan_loss,a_dis_loss+b_dis_loss))

                step += 1
            self.save_checkpoint(epoch+1, args)
            self.gen_scheduler.step()
            self.dis_scheduler.step()
            step = 0


        





