import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def image_generation(path, image_save_path, latent_size, eg_nos_latent = 100):

    """
    Generating the new fake data
    """

    model = torch.load(path, weights_only=False)

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    for j in range(10):
        folder = os.path.join(image_save_path, str(int(j)))
        if not os.path.exists(folder):
            os.makedirs(folder)
        z_test = torch.randn(eg_nos_latent, latent_size, 1, 1)
        target = torch.tensor([int(j)]).repeat(eg_nos_latent, 1)

        outputs_tensor = model.model_gen(z_test, target)
        outputs = outputs_tensor.detach().numpy()

        for i in range(outputs.shape[0]):
            save_path = os.path.join(folder, f"generated_image_{i}.png")
            
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(outputs[i].transpose(1, 2, 0) * 0.5 + 0.5)  # Unnormalize image
            plt.savefig(save_path)
            plt.close()

    return outputs_tensor