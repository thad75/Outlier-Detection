from lib.model_cifar import Encoder, Generator, Discriminator

import torch.optim as optim
import torch.nn as nn
import torch

def model_create(ngpu,latent_size,device):
    netE = Encoder(ngpu,latent_size).to(device)
    netG = Generator(ngpu,latent_size).to(device)
    netD = Discriminator(ngpu, latent_size).to(device)
    return netE, netG, netD

def train_settings(ngpu, latent_size, lr,device):
    # Training Loop
    alpha = 0.9
    model_create(ngpu,latent_size,lr)

    netE.apply(weights_init)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizerD = optim.Adam([{'params' : netE.parameters()},
                            {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))

    optimizerG = optim.Adam(netD.parameters(), lr = lr , betas = (0.5,0.999))

    return netE , netG, netD, optimizerD, optimizerG

  