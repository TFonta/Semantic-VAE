"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import models.networks as networks
import util.util as util
import numpy as np
import random
import time


class CA2SISModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE, self.netD_m = self.initialize_networks(opt)
        print("initialized models", flush = True)
        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)

        self.gen_parts = [1, 2, 4, 5, 6, 7, 10, 11, 12, 13]
        self.part2swap = 1

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, p = None, style_swap = False):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, data)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                # fake_image, _ = self.generate_fake(input_semantics, real_image)
                obj_dic = data['path']
                fake_image = self.save_style_codes(input_semantics, real_image, obj_dic)
            return fake_image
        elif mode == 'swap_part':
            with torch.no_grad():
                if input_semantics.size(0) > 16:
                    fake_image = self.generate_swapped(input_semantics[:16,], real_image[:16,], p, style_swap = style_swap)
                else:
                    fake_image = self.generate_swapped(input_semantics, real_image, p, style_swap = style_swap)
                return fake_image
        elif mode == 'swap_styles':
            with torch.no_grad():
                fake_image = self.generate_styles(input_semantics, real_image, p)
                return fake_image
        elif mode == 'parts_int':
            with torch.no_grad():
                fake_image = self.generate_parts_interpolation(input_semantics, real_image, p)
                return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netD = networks.define_D(opt) if opt.isTrain else None
        netG = networks.define_G(opt)
        netE = networks.define_E(opt) if opt.use_vae else None
        netD_m = networks.define_D_mask(opt) if opt.generate_masks and opt.isTrain else None
        
        if not opt.no_model_load:
            if not opt.isTrain or opt.continue_train:
                netG = util.load_network(netG, 'G', opt.which_epoch, opt)
                if opt.isTrain:
                    netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD, netE, netD_m

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda(non_blocking=True)
            data['instance'] = data['instance'].cuda(non_blocking=True)
            data['image'] = data['image'].cuda(non_blocking=True)

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()

        #print(input_label.size(), label_map.size(), label_map.max())
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image, data):
        G_losses = {}

        fake_image = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)
        fake_image_D = fake_image.clone()
    

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image_D, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)


        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():

            fake_image = self.generate_fake(input_semantics, real_image)
            fake_image_D = fake_image.clone()
        
            
            fake_image_D = fake_image_D.detach()
            fake_image_D.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image_D, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)
        

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):

        fake_image  = self.netG(real_image, input_semantics, input_semantics)

        return fake_image

###############################################################

    def save_style_codes(self, input_semantics, real_image, obj_dic):

        fake_image = self.netG(real_image, input_semantics, input_semantics)
        
        fake_out = {'real':real_image,
                    'fake':fake_image
                    }
        return fake_out

    def generate_swapped(self, input_semantics, real_image, p, style_swap = False):

        input_semantics_sw, _ = self.swap_parts(input_semantics, p, randp=False, k=len(p))
        fake_image_sw = self.netG(real_image, input_semantics, input_semantics_sw)
        
        if not style_swap:
            start_time = time.time()
            fake_image = self.netG(real_image, input_semantics, input_semantics)  
            #print("--- %s seconds ---" % (time.time() - start_time))      
        else:
            # invert real images order
            first_half = real_image[:real_image.size(0)//2].clone()
            second_half = real_image[real_image.size(0)//2:].clone()

            real_image[:real_image.size(0)//2] = second_half
            real_image[real_image.size(0)//2:] = first_half

            # invert input semantcs order
            input_semantics_styles_sw = input_semantics.clone()

            first_half = input_semantics[:real_image.size(0)//2].clone()
            second_half = input_semantics[real_image.size(0)//2:].clone()

            input_semantics_styles_sw[:real_image.size(0)//2] = second_half
            input_semantics_styles_sw[real_image.size(0)//2:] = first_half

            fake_image = self.netG(real_image, input_semantics_styles_sw, input_semantics) 

        fake_out = {'real':real_image,
                    'fake':fake_image,
                    'fake_sw':fake_image_sw}

        return fake_out #real_image, fake_image, fake_image_sw, p

    def generate_parts_interpolation(self, input_semantics, real_image, p):

        input_semantics_sw, _ = self.swap_parts(input_semantics, p, randp=False, k=len(p))
        fake_image_sw = self.netG(real_image, input_semantics, input_semantics_sw)
        
        s = self.netG.style_encoder(real_image, input_semantics)
        z = self.netG.encode(input_semantics[:,1:])
        z_sw = self.netG.encode(input_semantics_sw[:,1:])
        z_int = z.clone()

        fake_image = self.netG.decode(z,s)
        fake_image_sw = self.netG.decode(z_sw,s)
        fake_int = []
        
       #alpha = torch.tensor([0.,0.25,0.5,0.75,1.]).cuda()      
        alpha = torch.tensor([1.,0.75,0.5,0.25,0.]).cuda()  
        p = [el - 1 for el in p]  
        for a in alpha:
            int_z = a*z[:,p] + (1-a)*z_sw[:,p]
            z_int[:,p] = int_z
            fake_int.append(self.netG.decode(z_int,s))
        fake_int = torch.stack(fake_int)

        
        # invert input semantcs order
        input_semantics_styles_sw = input_semantics.clone()

        first_half = input_semantics[:real_image.size(0)//2].clone()
        second_half = input_semantics[real_image.size(0)//2:].clone()

        input_semantics_styles_sw[:real_image.size(0)//2] = second_half
        input_semantics_styles_sw[real_image.size(0)//2:] = first_half


        z_full_sw = self.netG.encode(input_semantics_styles_sw[:,1:])
        z_full_int = z.clone()
        fake_full_int = []
        for a in alpha:
            int_z = a*z + (1-a)*z_full_sw
            z_full_int = int_z
            fake_full_int.append(self.netG.decode(z_full_int,s))
        fake_full_int = torch.stack(fake_full_int)

        fake_out = {'real':real_image,
                    'fake':fake_image,
                    'fake_sw':fake_image_sw,
                    'fake_int':fake_int,
                    'fake_full_int':fake_full_int}

        return fake_out
    
    def generate_styles(self, input_semantics, real_image, p, level = None):
        
        m = input_semantics
        if self.opt.exclude_bg:
            m_sw = input_semantics[:,1:]
        else:
            m_sw = input_semantics
        
        z = self.netG.encode(m_sw)
                    
        s_org = self.netG.style_encoder(real_image, m)

        s_swap = s_org.clone()

        first_half = s_org[:real_image.size(0)//2].clone()
        second_half = s_org[real_image.size(0)//2:].clone()
        s_swap[real_image.size(0)//2:] = first_half
        s_swap[:real_image.size(0)//2] = second_half

        for part in p:
            s_org[:,part] = s_swap[:,part].clone()  
            
        fake_image = self.netG.decode(z,s_org)
        
        fake_image_sw = self.netG.decode(z,s_swap)

        fake_out = {'real':real_image,
                    'fake':fake_image,
                    'fake_sw': fake_image_sw}

        return fake_out

    def add_background_channel(self, x):
        bg_mask = torch.sum(x, dim = 1) > 0
        bg_mask = 1. - bg_mask.unsqueeze(1).float()
        return torch.cat((bg_mask, x), dim = 1)


    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def discriminate_mask(self, fake_image, real_image):
        fake_and_real = torch.cat([fake_image, real_image], dim=0)

        discriminator_out = self.netD_m(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


    def swap_parts(self, input_semantics, p, randp=False, k=1):
        # Select existing part in both images
        sumParts = np.sum(input_semantics.cpu().numpy(), axis=(2, 3))
        input_semantics_sw = input_semantics.clone()
        
        half_batch = input_semantics.size(0)//2

        swap_label = torch.zeros(input_semantics.size(0), input_semantics.size(1)).cuda()
        for el_idx in range(half_batch):
            commonParts  = np.all([sumParts[el_idx, :], sumParts[el_idx + half_batch, :]], axis=0)
            
            idx = [i for i, x in enumerate(commonParts) if x]
            if randp:
                pToSwap = random.choices(idx, k=k)
            else:
                pToSwap = list(set(p) & set(idx))

            
            for el in pToSwap:
                swap_label[el_idx,el] = 1.
                swap_label[el_idx + half_batch,el] = 1.
            
            if len(pToSwap) > 0:
                input_semantics_sw[el_idx, pToSwap, :, :] = input_semantics[el_idx + half_batch, pToSwap, :, :]
                input_semantics_sw[el_idx + half_batch, pToSwap, :, :] = input_semantics[el_idx, pToSwap, :, :]

        return input_semantics_sw, swap_label