import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_3D
from util.visualizer import test_saving
import numpy as np

class AGANForegroundModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G', 'G_A', 'G_B', 'D_A', 'D_B', 'D_C', 'idt_A', 'idt_B', 'cycle_A', 'cycle_B']
        self.loss_names = ['G', 'G_A', 'G_B', 'D_A', 'D_B', 'idt_A', 'idt_B', 'cycle_A', 'cycle_B'] # yb
        # self.loss_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_C', 'idt_A', 'idt_B', 'cycle_A', 'cycle_B', 'G_A1','G_A2'] # yb
        # self.loss_names = ['G', 'G_A', 'G_B', 'D_A', 'D_B', 'D_C', 'idt_A', 'idt_B', 'idt_C', 'cycle_A', 'cycle_B', 'cycle_C'] # yb
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # visual_names_A = ['real_A', 'fake_B', 'rec_A', 'fake_C', 'fore_b', 'back_b', 'o1_b', 'o2_b', 'o3_b', 'o4_b', 'o5_b', 'o6_b', 'o7_b', 'o8_b', 'o9_b', 'o10_b',
        # 'a1_b', 'a2_b', 'a3_b', 'a4_b', 'a5_b', 'a6_b', 'a7_b', 'a8_b', 'a9_b', 'a10_b', 'i1_b', 'i2_b', 'i3_b', 'i4_b', 'i5_b', 
        # 'i6_b', 'i7_b', 'i8_b', 'i9_b']
        # visual_names_A = ['real_A', 'fake_B', 'rec_A', 'fake_C', 'fore_b', 'back_b']
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'fore_b', 'back_b'] # yb
        # visual_names_B = ['real_B', 'fake_A', 'rec_B', 'fake_C', 'fore_a', 'back_a', 'o1_a', 'o2_a', 'o3_a', 'o4_a', 'o5_a', 'o6_a', 'o7_a', 'o8_a', 'o9_a', 'o10_a', 
        # 'a1_a', 'a2_a', 'a3_a', 'a4_a', 'a5_a', 'a6_a', 'a7_a', 'a8_a', 'a9_a', 'a10_a', 'i1_a', 'i2_a', 'i3_a', 'i4_a', 'i5_a', 
        # 'i6_a', 'i7_a', 'i8_a', 'i9_a']
        # visual_names_B = ['real_B', 'fake_A', 'rec_B', 'real_C', 'fore_a', 'back_a']
        # visual_names_B = ['real_B', 'fake_A', 'rec_B', 'rec_C', 'real_C', 'fore_a', 'back_a'] # yb
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'fore_a', 'back_a'] # yb
        

        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
            # visual_names_A.append('idt_C') # yb

        if self.opt.saveDisk:
            self.visual_names = ['real_A', 'fake_B', 'a10_b', 'real_B','fake_A', 'a10_a']
        else:
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks_3D> and <BaseModel.load_networks_3D>.
        if self.isTrain:
            # self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_C']
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B'] # yb
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks_3D (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_A = networks_3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'our', opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device) # yb 
        # self.netG_B = networks_3D.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'our', opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device) # yb 
        
        # yb for gpu test
        self.netG_A = networks_3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, 'A', self.gpu_ids).to(self.device)
        self.netG_B = networks_3D.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, 'B', self.gpu_ids).to(self.device)

        if self.isTrain:  # define discriminators
            # self.netD_A = networks_3D.define_D(opt.output_nc, opt.ndf, opt.netD, 'D_A',
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)  # yb netD_name
            # self.netD_B = networks_3D.define_D(opt.input_nc, opt.ndf, opt.netD, 'D_B',
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)  # yb netD_name
            
            # yb for gpu test
            self.netD_A = networks_3D.define_D(opt.output_nc, opt.ndf, opt.netD, 'D_A',
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, 'A', self.gpu_ids).to(self.device)
            self.netD_B = networks_3D.define_D(opt.input_nc, opt.ndf, opt.netD, 'D_B',
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, 'B', self.gpu_ids).to(self.device)

            #NOTE: Discriminator for Lesion Masks
            # self.netD_C = networks_3D.define_D(opt.input_nc, opt.ndf, opt.netD, 'D_C',
                                # opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # yb netD_name

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.fake_C_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks_3D.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            #NOTE: added netD_L to optimizer_D
            # self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_C.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999)) # yb
            # self.optimizer_G = torch.optim.Adagrad(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, ) #.#
            # self.optimizer_D = torch.optim.Adagrad(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_C.parameters()), lr=opt.lr, ) #.#
            
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.real_C  = input['C'].to(self.device)
        # C_name = "C_1"
        test_saving(self.real_A, 'just_before_generator')
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        self.fake_B, self.fore_b, self.back_b, self.o1_b, self.o2_b, self.o3_b, self.o4_b, self.o5_b, self.o6_b, self.o7_b, self.o8_b, self.o9_b, self.o10_b, \
        self.a1_b, self.a2_b, self.a3_b, self.a4_b, self.a5_b, self.a6_b, self.a7_b, self.a8_b, self.a9_b, self.a10_b, \
        self.i1_b, self.i2_b, self.i3_b, self.i4_b, self.i5_b, self.i6_b, self.i7_b, self.i8_b, self.i9_b = self.netG_A(self.real_A, 'G_A')  # G_A(A)

        self.rec_A, _, _, _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _ = self.netG_B(self.fake_B, 'G_B')   # G_B(G_A(A))

        self.fake_A, self.fore_a, self.back_a, self.o1_a, self.o2_a, self.o3_a, self.o4_a, self.o5_a, self.o6_a, self.o7_a, self.o8_a, self.o9_a, self.o10_a, \
        self.a1_a, self.a2_a, self.a3_a, self.a4_a, self.a5_a, self.a6_a, self.a7_a, self.a8_a, self.a9_a, self.a10_a, \
        self.i1_a, self.i2_a, self.i3_a, self.i4_a, self.i5_a, self.i6_a, self.i7_a, self.i8_a, self.i9_a = self.netG_B(self.real_B, 'G_B')  # G_B(B)
        
        # self.fake_C = self.fore_b       # Using the foreground output as discriminator against real lesion mask
        self.rec_B, _, _, _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _ = self.netG_A(self.fake_A, 'G_A')   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D


    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    #NOTE: added discriminator for fake lesion mask
    def backward_D_C(self):
        """Calculate GAN loss for discriminator D_C"""
        fake_C = self.fake_C_pool.query(self.fake_C)
        # fake_name1 = "C_fake1"
        # fake_name2 = "C_fake2"
        # test_saving(fake_C, fake_name1)
        # test_saving(self.fake_C, fake_name2)
        self.loss_D_C = self.backward_D_basic(self.netD_C, self.real_C, fake_C)
        
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # yb
            self.idt_A, self.idt_C, _, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _  = self.netG_A(self.real_B, 'G_A')
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # self.loss_idt_C = self.criterionIdt(self.idt_C, self.real_C) * lambda_B * lambda_idt # yb
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _, _, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _  = self.netG_B(self.real_A, 'G_B')
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            # self.loss_idt_C = 0 # yb

        # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B) + self.netD_C(self.fake_C), True)    #default    # here we feed in the discriminator of lesions
        # self.new_fake_B = self.fake_B[-1,-1,:,:,:]
        # self.new_fake_A = self.fake_A[-1,-1,:,:,:]
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) # test yb
        # self.loss_G_A = self.criterionGAN(0.4*(self.netD_A(self.fake_B)) + 0.6*(self.netD_C(self.fake_C)), True)
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B),True) + self.criterionGAN(self.netD_C(self.fake_C), True)  ##HH
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B  
        # Backward cycle loss of ROI
        # self.loss_cycle_C = self.criterionCycle(self.rec_C, self.real_C) * lambda_B  # yb                       
        # combined loss and calculate gradients NOTE: maybe play around with the thing below.
        self.loss_G = self.loss_G_A + self.loss_G_B  + self.loss_idt_A + self.loss_idt_B + self.loss_cycle_A + self.loss_cycle_B
        # self.loss_G = self.loss_G_A + self.loss_G_B  + self.loss_idt_A + self.loss_idt_B + (self.loss_idt_C*0.3) + self.loss_cycle_A + self.loss_cycle_B + (self.loss_cycle_C*0.3) # yb
        self.loss_G.backward()
        
        # self.loss_G_A1 = self.criterionGAN(self.netD_A(self.fake_B), True) # yb
        # self.loss_G_A2 = self.criterionGAN(self.netD_C(self.fake_C), True) # yb

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        # self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A, self.netD_B], False) # yb
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B and D_C
        # self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C], True)
        self.set_requires_grad([self.netD_A, self.netD_B], True) # yb
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        # self.backward_D_C()      #calculate gradients for D_C
        self.optimizer_D.step()  # update D_A and D_B's weights
        
