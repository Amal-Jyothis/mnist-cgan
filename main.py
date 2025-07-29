import time
import datetime

from packages.data_collect import*
from packages.c_gan import*
from packages.img_generation import*


def main():
    start_time = time.time()
    print('Run started at:', datetime.datetime.now())

    '''
    Start of input data extraction
    '''
    dataloader = input_data()

    '''
    Train gan model
    '''
    model_save_path = r'output/saved_model/cgan_model.pth'
    image_save_path = r'output/generated_images/'
    hyperparameters = {'learning_rate_G': 5e-4,
                       'learning_rate_D': 5e-4,
                       'g_iter': 1,
                       'd_iter': 5,
                       'latent_size': 20,
                       'reg_G': 0,
                       'reg_D': 0}

    cgan(dataloader, model_save_path, image_save_path, **hyperparameters)

    image_generation(model_save_path, image_save_path, hyperparameters['latent_size'], eg_nos_latent=100)

    end_time = time.time()
    print('Time taken:', end_time - start_time)

if __name__ == "__main__":
    main()
    