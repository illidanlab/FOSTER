import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time

from models.wrn_virtual import oodGenerator, InversGenerator
MIN_SAMPLES_PER_LABEL=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#central generator using crossentropy
class CentralGen:
    def __init__(self, args, local_iter, num_class, model='wrn'):
        self.num_class = num_class
        self.local_iter = local_iter
        self.generative_model = InversGenerator(num_class, args.widen_factor, width_scale=args.width_scale, model=model).to(device)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=1e-4, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=1e-2, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)


    def train_generator(self, args, user_classifier, epoches=1):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param net: local training model
        :param train_loader: local training loader (ID data)
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return angeneratorything.
        """
        #self.generative_regularizer.train()
        DIVERSITY_LOSS = 0

        def update_generator_(args, diversity_loss):
            self.generative_model.train()
            for i in range(self.local_iter):
                self.generative_optimizer.zero_grad()
                oodclass = [i for i in range(self.num_class)]
                oody = np.random.choice(oodclass, args.oe_batch_size)
                oody = torch.LongTensor(oody).cuda()
                oodz = self.generative_model(oody)
                logit_given_gen = user_classifier(oodz)
                if args.method == 'crossentropy':
                    diversity_loss = F.cross_entropy(logit_given_gen, oody)  # encourage different outputs


                diversity_loss.backward()
                self.generative_optimizer.step()
            return diversity_loss
        for i in range(epoches):
            DIVERSITY_LOSS=update_generator_(
               args, DIVERSITY_LOSS)
        info="Generator:  Diversity Loss = {:.4f}, ". \
            format(DIVERSITY_LOSS)
        print(info)
        self.generative_lr_scheduler.step()