import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        #self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        #self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        # see https://gist.github.com/SsnL/351720fb0fd0a43c6fdc370be402cff3
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/170

        # flag for indicating usage of Wasserstein GAN
        self.use_wgan = opt.wgan
        self.wgan_n_critic = opt.wgan_n_critic
        self.wgan_clamp_lower = opt.wgan_clamp_lower
        self.wgan_clamp_upper = opt.wgan_clamp_upper
        self.wgan_train_critics = False

        if self.use_wgan:
            print("We are Wasserstein-GANing")

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                     opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                    opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)

        if self.isTrain: # if not trained
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train: # already trained
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr # learning rate
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.wgan:
                self.criterionGAN = networks.WassersteinGANLoss() # WGAN loss
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers

            if opt.wgan:
                if opt.wgan_optimizer == 'adam':
                    self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                        lr=opt.wgan_lrG, betas=(0.5, 0.9))
                    self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.wgan_lrD, betas=(0.5, 0.9))
                    self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.wgan_lrD, betas=(0.5, 0.9))
                else:
                    self.optimizer_G = torch.optim.RMSprop(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                           lr=opt.wgan_lrG)
                    self.optimizer_D_A = torch.optim.RMSprop(self.netD_A.parameters(), lr=opt.wgan_lrD)
                    self.optimizer_D_B = torch.optim.RMSprop(self.netD_B.parameters(), lr=opt.wgan_lrD)
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            networks.print_network(self.netG_B)
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        #self.input_A.resize_(input_A.size()).copy_(input_A)
        #self.input_B.resize_(input_B.size()).copy_(input_B)
        # See https://gist.github.com/SsnL/351720fb0fd0a43c6fdc370be402cff3
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(fake_A).data
        self.fake_A = fake_A.data

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        # next line does not work
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_wasserstein(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake)
        loss_D = self.criterionGAN(pred_fake, pred_real, generator_loss=False)
        #loss_D = -(pred_real.mean() - pred_fake.mean())
        # D wants to max pred_real.mean() and min pred_fake.mean()
        # so D wants to min -pred_real.mean() + pred_fake.mean()
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data[0]

    def backward_D_A_wgan(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_wasserstein(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]
        #self.loss_D_A.backward(retain_variables=True)

    def backward_D_B_wgan(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_wasserstein(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data[0]
        #self.loss_D_B.backward(retain_variables=True)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        fake_B = self.netG_A.forward(self.real_A)
        pred_fake = self.netD_A.forward(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netG_B.forward(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        rec_B = self.netG_A.forward(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]

    def backward_wgan_G(self, do_backward=True):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Wasserstein-GAN loss
        # G_A(A)
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, generator_loss=True)
        # loss_G_A = -pred_fake.mean()
        # G_A wants to max pred_fake (i.e. min -pred_fake)

        # G_B(B)
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, generator_loss=True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        # Combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]

        #if do_backward:
        #    # Backprop
        #    self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        if self.use_wgan:
            if self.wgan_train_critics:
                # Train the critics to optimality
                for i_critic in range(self.wgan_n_critic):
                    # Clip the parameters for k-Lipschitz continuity
                    for p in self.netD_A.parameters():
                        p.data.clamp_(self.wgan_clamp_lower, self.wgan_clamp_upper)
                    for p in self.netD_B.parameters():
                        p.data.clamp_(self.wgan_clamp_lower, self.wgan_clamp_upper)
                    self.optimizer_D_A.zero_grad()
                    self.optimizer_D_B.zero_grad()
                    self.backward_D_A_wgan()
                    self.backward_D_B_wgan()
                    #self.backward_wgan_D()
                    self.optimizer_D_A.step()
                    self.optimizer_D_B.step()

            # Train the generators
            self.optimizer_G.zero_grad()
            self.backward_wgan_G()
            self.optimizer_G.step()

            #self.backward_wgan_G(do_backward=False)
            #if self.wgan_train_critics:
            #    self.optimizer_G.step()
            #else:
            #    self.wgan_train_critics = True

        else:
            # G_A and G_B
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            # D_A
            self.optimizer_D_A.zero_grad()
            self.backward_D_A()
            self.optimizer_D_A.step()
            # D_B
            self.optimizer_D_B.zero_grad()
            self.backward_D_B()
            self.optimizer_D_B.step()


    def get_current_errors(self):
        D_A = self.loss_D_A.data[0]
        G_A = self.loss_G_A.data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        D_B = self.loss_D_B.data[0]
        G_B = self.loss_G_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B)
        rec_A  = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A)
        rec_B  = util.tensor2im(self.rec_B)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A)
            idt_B = util.tensor2im(self.idt_B)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
