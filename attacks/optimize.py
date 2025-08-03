from losses.poincare import poincare_loss
import math

import numpy as np
import torch
import torch.nn as nn


from utils.stylegan import save_images_2
class Optimization():
    def __init__(self, target_model, synthesis, discriminator, transformations, num_ws, config):
        self.synthesis = synthesis
        self.target = target_model
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.num_ws = num_ws
        self.clip = config.attack['clip']

    def optimize(self, w_batch, targets_batch, num_epochs,idx=0):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        # print('scheduler',scheduler)
        # exit()
        all_w_batch = []
        
        all_w_batch.append(w_batch.detach())
        # with torch.no_grad():
        #     imgs = self.synthesize(w_batch, num_ws=self.num_ws)
        #     save_images_2(imgs,'results_images2/','{}_e_{}'.format(idx,0))
        # Start optimization
        all_loss = []
        grad =[]
        lrs = []
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(
                    imgs)
            else:
                discriminator_loss = torch.tensor(0.0)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(
                outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight
            loss.backward()
            all_loss.append(loss.item())
                    
            # print('w',w.grad.norm())
            # print('w.grad',w_batch.grad.shape)
            # exit()
            grad.append(w_batch.grad.detach().clone().norm().item())
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()

            if scheduler:
                scheduler.step()
            all_w_batch.append(w_batch.detach().cpu())
            # with torch.no_grad():
            #     imgs = self.synthesize(w_batch, num_ws=self.num_ws)
            #     save_images_2(imgs,'results_images2/','{}_e_{}'.format(idx,i+1))

            # print('all_w_batch',i-1,all_w_batch[i-1][:5])
            # print('all_w_batch',i,all_w_batch[i][:5])
            # Log results
            if self.config.log_progress:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(
                        confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()

                if torch.cuda.current_device() == 0:
                    print(
                        f'iteration {i}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f}'
                    )
        
        import matplotlib.pyplot as plt
        file_path = './loss_grad/'
        fig, ax1 = plt.subplots(figsize=(8,5))


        # First axis for gradient norm
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(range(num_epochs), all_loss, color=color, marker='o', label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        # ax1.set_ylim(0, 20)  # Notice: larger value at top

        # Second axis for learning rate
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Learning Rate', color=color)
        ax2.plot(range(num_epochs), lrs, color=color, label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor=color)
        # ax2.set_ylim(0.0005, 0.005)  # lr range from 0.0005 to 0.005



        # plt.plot(all_loss, marker='o', linestyle='-', color='b')
        # plt.plot(lrs)
        plt.title("Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # # plt.grid(True)

        # Save the plot to a file
        plt.savefig(f"{file_path}loss_{targets_batch[0]}.png")
        plt.close()

        fig, ax1 = plt.subplots(figsize=(8,5))


        # First axis for gradient norm
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Norm', color=color)
        ax1.plot(range(num_epochs), grad, color=color, marker='o', label='Gradient Norm')
        ax1.tick_params(axis='y', labelcolor=color)
        # ax1.set_ylim(0, 20)  # Notice: larger value at top

        # Second axis for learning rate
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Learning Rate', color=color)
        ax2.plot(range(num_epochs), lrs, color=color, label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor=color)
        # ax2.set_ylim(0.0005, 0.005)  # lr range from 0.0005 to 0.005



        # plt.figure(figsize=(8,5))
        # print('grad',len(grad))
        # plt.plot(range(n_epochs), grad, marker='o')
        # plt.plot(lrs)
        plt.title('Gradient Norm per Iteration')
        # plt.xlabel('Iteration')
        # plt.ylabel('Gradient Norm (L2)')
        # plt.grid(True)
        
        # Save the plot to a file
        plt.savefig(f"{file_path}grad_{targets_batch[0]}.png")
        plt.close()
        return w_batch.detach(), all_w_batch

    def synthesize(self, w, num_ws):
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            imgs = self.synthesis(w_expanded,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis(w, noise_mode='const', force_fp32=True)
        return imgs

    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def compute_discriminator_loss(self, imgs):
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(
            -discriminator_logits).mean()
        return discriminator_loss
