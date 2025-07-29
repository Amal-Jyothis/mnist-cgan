import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class Generator(torch.nn.Module):
    """
    Generator definition
    """
    def __init__(self, latent_size):
        super(Generator, self).__init__()

        self.embedding = torch.nn.Embedding(10, latent_size)
        
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_size*2, 32, 4, 1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(32, 16, 5, 1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(16, 8, 5, 1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(8, 1, 5, 1, bias=False),
            torch.nn.Tanh()
        )
    
    def forward(self, input, target):
        enc = self.embedding(target)
        enc = enc.view(-1, enc.size()[2], 1, 1)
        input = torch.cat((input, enc), 1)
        # input = input + enc
        output = self.main(input)
        return output
    
class Discriminator(torch.nn.Module):
    """
    Discriminator definition
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embedding = torch.nn.Embedding(10, 256)
        
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(2, 8, 5, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(8, 32, 5, 1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(32, 8, 5, 1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(8, 1, 4, 1, bias=False),
            torch.nn.Sigmoid()    
        )
    
    def forward(self, input, target):
        enc = self.embedding(target)
        enc = torch.reshape(enc, (enc.size()[0], 1, 16, 16))
        input = torch.cat((input, enc), 1)
        output = self.main(input)
        return output

class model_definition():
    """
    This class defines the optimiser type and the learning rate used for optimization
    """
    def __init__(self, latent_size, learning_rate_G, learning_rate_D, reg_G, reg_D, beta_1=0.9, beta_2=0.999):
        betas = (beta_1, beta_2)

        self.model_gen = Generator(latent_size)
        self.optimizerG = torch.optim.RMSprop(self.model_gen.parameters(), lr=learning_rate_G, weight_decay=reg_G)

        self.model_discr = Discriminator()
        self.optimizerD = torch.optim.RMSprop(self.model_discr.parameters(), lr=learning_rate_D, weight_decay=reg_D)

def cgan(dataloader, model_save_path, image_save_path, **kwargs):
    """
    This function trains a GAN model with the given training data and hyperparameters.

    Parameters:
    x_train: Training data
    **kwargs: Additional hyperparameters
    """

    # define required hyperparameters
    lr_G = float(kwargs.get("learning_rate_G"))
    lr_D = float(kwargs.get("learning_rate_D"))
    g_iter = int(kwargs.get("g_iter"))
    d_iter = int(kwargs.get("d_iter"))
    latent_size = int(kwargs.get("latent_size"))
    reg_G = float(kwargs.get("reg_G"))
    reg_D = float(kwargs.get("reg_D"))

    model = model_definition(latent_size, lr_G, lr_D, reg_G, reg_D)

    print('Training c-GAN model...')
    training_cgan(model, dataloader, latent_size, num_epochs=100, discr_train_iter = d_iter, gen_train_iter = g_iter)

    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    torch.save(model, model_save_path)

    return 


"""
training model
"""
def training_cgan(model, dataloader, latent_size, num_epochs = 5, discr_train_iter = 5, gen_train_iter = 1):

    """"
    Initialize loss values for plotting
    """
    loss_plotD = np.zeros(num_epochs)
    loss_plotGD = np.zeros(num_epochs)
    loss_plot_totalD = np.zeros(num_epochs)
    loss_plotG = np.zeros(num_epochs)


    epochs = np.arange(0, num_epochs)

    for epoch in range(num_epochs):
        for i, (images, target) in enumerate(dataloader):
                
            """"" 
            Training discriminator
            """

            for i in range(0, discr_train_iter):
                model.optimizerD.zero_grad()

                """"
                Calculating D(X) and loss function
                """

                target = target.to(torch.int).view(-1, 1)

                outputs_1 = model.model_discr(images, target).view(-1, 1)
                y_train = torch.full((images.size()[0], 1), 1.0)

                """"
                Binary cross entropy loss for discriminator
                """
                loss_frm_D = torch.nn.BCELoss()(outputs_1, y_train)
                

                """"
                Calculating D(G(z)) and loss function
                """

                z = torch.randn(images.size()[0], latent_size, 1, 1)
                gen_output_1 = model.model_gen(z, target).detach()
                outputs_2 = model.model_discr(gen_output_1, target).view(-1, 1)
                z_output = torch.full((images.size()[0], 1), 0.0)

                """"
                Binary cross entropy loss for discriminator
                """
                loss_frm_GD = torch.nn.BCELoss()(outputs_2, z_output)               
                
                total_loss = loss_frm_D + loss_frm_GD
                total_loss.backward()
                model.optimizerD.step()

            loss_plotD[epoch] = loss_frm_D
            loss_plotGD[epoch] = loss_frm_GD
            loss_plot_totalD[epoch] = total_loss

            """
            Training generator
            """

            for j in range(0, gen_train_iter):
                """
                Calculating D(G(z)) and training
                """

                model.optimizerG.zero_grad()
                z = torch.randn(images.size()[0], latent_size, 1, 1)
                # z = torch.cat((z, target_mod_gen), 1)
                gen_output = model.model_gen(z, target)
                # gen_output = torch.cat((gen_output, target_mod_discr), 1)
                outputs = model.model_discr(gen_output, target).view(-1, 1)
                output_label = torch.full((images.size()[0], 1), 1.0)

                '''
                BCE loss for Generator
                '''
                loss_frm_G = torch.nn.BCELoss()(outputs, output_label)
                
                loss_frm_G.backward()
                model.optimizerG.step()

            loss_plotG[epoch] += loss_frm_G


    """
    Plotting
    """
    print('Loss of Generator: ', loss_plotG[num_epochs - 1])
    print('Loss of Discriminator: ', loss_plot_totalD[num_epochs - 1])

    plt.plot(epochs, loss_plotD, label='Discriminator Loss for real')
    plt.plot(epochs, loss_plotGD, label='Discriminator Loss for generated')
    plt.plot(epochs, loss_plot_totalD, label='Discriminator Total Loss')
    plt.plot(epochs, loss_plotG, label='Generator Loss')
    plt.tick_params(axis='both', labelsize=10)
    plt.xlabel('Iterations', fontsize='10')
    plt.ylabel('Loss', fontsize='10')
    plt.legend(fontsize='10')
    plt.grid()
    plt.savefig(r'cGAN_loss_plot.png', dpi=1000)
    # plt.show()
