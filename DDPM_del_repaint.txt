import torch
from torch.nn import functional as F
import torch.nn as nn

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from skimage import io
from dataclasses import dataclass
from typing import List

#%%

def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        
        self.register_buffer( 'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, label1, label2):

        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise

        if label1 !=None:
            if label2 == None:
                loss = F.mse_loss(self.model(x_t, t, label1, None), noise, reduction='none')
            else:
                loss = F.mse_loss(self.model(x_t, t, label1, label2), noise, reduction='none')
        else:
            loss = F.mse_loss(self.model(x_t, t, None, None), noise, reduction='none')

        return loss
    
@dataclass
class CustomScheduler:
    def __init__(self, T: torch.Tensor, betas: torch.Tensor):
        # assert len(T) == len(betas)
        self.T = T
        self.timesteps = torch.arange(0, eval_config["T"], dtype=torch.long, device=eval_config ["device"])
        self.betas = betas
        # TODO verify this
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.roll(self.alphas_cumprod, 1)
        self.alphas_cumprod_prev[0] = 1.0

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    @classmethod
    def from_DDPMScheduler(cls, ddpm_scheduler):
        return cls(ddpm_scheduler.T, ddpm_scheduler.betas)

@torch.no_grad()
def single_reverse_step(model, x: torch.Tensor, t: int, S: CustomScheduler) -> torch.Tensor:
    """
    applies the model to go from timestep t to t-1
    See Algorithm 2 of https://arxiv.org/pdf/2006.11239.pdf
    :param model: the model that predicts the noise
    :param x: the data at timestep t
    :param t: the current timestep
    :param scheduler: class that provides the variance schedule
    :return: the data at diffusion timestep t-1
    """

    mean = S.sqrt_recip_alphas[t] * (x - S.betas[t] * model(x, x.new_ones([x.shape[0], ], dtype=torch.long) * t, None, None) / S.sqrt_one_minus_alphas_cumprod[t])
    if t == 0:
        return mean
    else:
        noise = torch.randn_like(x) * torch.sqrt(S.posterior_variance[t])
        return mean + noise

@torch.no_grad()
def zero_to_t(x_0: torch.Tensor, t: int, S: CustomScheduler) -> torch.Tensor:
    if t == 0:
        return x_0
    else:
        return torch.sqrt(S.alphas_cumprod[t]) * x_0 + \
                torch.sqrt(1.0 - S.alphas_cumprod[t]) * torch.randn_like(x_0)

@torch.no_grad()
def forward_j_steps(x_t: torch.Tensor, t: int, j: int, S: CustomScheduler)-> torch.Tensor:
    partial_alpha_cumprod = S.alphas_cumprod[t+j]/S.alphas_cumprod[t]
    return torch.sqrt(partial_alpha_cumprod) * x_t + \
            torch.sqrt(1.0 - partial_alpha_cumprod) * torch.randn_like(x_t)

def get_jumps(timesteps, jumps_every:int=100, r:int=5) -> List[int]:
    jumps = []
    for i in range(0, torch.max(timesteps), jumps_every):
        jumps.extend([i] * r)
    jumps.reverse()  # must be in descending order
    return jumps

@torch.no_grad()
def repaint(original_data: torch.Tensor, 
            keep_mask: torch.Tensor,
            model, 
            scheduler: CustomScheduler, 
            j:int=10, 
            r:int=20) -> torch.Tensor:
    """
    repaints that which isn't in the mask using the provided diffusion model
    :param original_image: the original data to repaint. Must be in the range that the model expects (usually [-1, 1])
    :param keep_mask: the mask of the image to keep. All values must be 0 or 1
    :param model: the diffusion model to use
    :param scheduler: the scheduler to use, must be compatible with the model
    """

    jumps = get_jumps(scheduler.timesteps, r=r)

    device = original_data.device
    sample = torch.randn_like(original_data).to(device) # sample is x_t+1
    print("beginning repaint")
    for t in tqdm(scheduler.timesteps):

        # the following loop handles the bouts of resampling
        while len(jumps) > 0 and jumps[0] == t:
            jumps = jumps[1:]
            sample = forward_j_steps(sample, t, j, scheduler)
            # after the resample, come back down to the current timestep
            for override_t in range(t + j, t, -1):
                sample = single_reverse_step(model, sample, override_t, scheduler)

        x_known = zero_to_t(original_data, t, scheduler)
        x_unknown = single_reverse_step(model, sample, t, scheduler)
        sample = keep_mask * x_known + (1-keep_mask) * x_unknown
        if t % 50 == 0:
                center_slice = x_unknown[0, 0, :, :, x_unknown.shape[-1] // 2].cpu()
                plt.imshow(center_slice, cmap="gray")
                plt.title(f"Timestep {t}")
                plt.pause(0.1)
    return sample   


def initialize_model(model_config, device):

    from UNET import Conditional_UNet
    unet_model_config = {"T": 1000,
                         "ch" : 32,
                         "ch_mult" : [1, 2, 2, 4],
                         "num_res_blocks" : 1, #432 ( FOR CAT CASE) #216 #(6)**3#768
                         "dropout" : 0.15,
                         }
    
    model = Conditional_UNet(T=unet_model_config["T"], 
                             ch=unet_model_config["ch"], 
                             ch_mult=unet_model_config["ch_mult"],
                             num_res_blocks=unet_model_config["num_res_blocks"], 
                             dropout=unet_model_config["dropout"],
                             
                             img_channels=model_config["img_channels"]).to(device)
    # model.apply(weights_init_normal3d)
    # print(model)
    if model_config.get("training_load_weight"):
        print("weight loaded")
        model.load_state_dict(torch.load(os.path.join(model_config["save_dir"], model_config["training_load_weight"]), map_location=device))
    return model




if __name__ == "__main__":
    eval_config = {
       
       # Prepare DDPM
       "T": 1000,
       "beta_1": 1e-4,
       "beta_T": 0.02,
       "sampling_type": "ddpm",
       
       # Prepare DDIM specific params
       "beta_schedule": "linear",
       "ddim_steps": 1000,
       "eta": 0,
       
       # prepare eval parameters
       "num_runs": 1,
       "eval_batch_size": 10,
       "eval_img_size": 64,
    
       # Prepare extra params
       "path_ckpt": "./train_checkpoints/ddpm_checkpoint_epoch_1200.pt",
       "path_save_outputs": "./eval_samples_inpainting",
       "device": "cuda:1",
       "seed": None,
      
       # Prepare other params for lddpm and ddpm
       "img_channels": 1, # image (1) or latent channels (4)
       "path_ckpt_vae":"",
       "training_type":"ddpm",
       "model_type": "unet", # or "dit"
       "scale_latents": "",
    
       }
        
    image_path = "eval_samples/generated_images_0_1.tif"
    
    image = io.imread(image_path)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(1).to(eval_config ["device"])
    mask = torch.ones(*image.shape).to(eval_config ["device"]) # 1--> keep, 0-->inpaint
    mask[:, :, 
          16:image.shape[-3]-16, 
          16:image.shape[-2]-16, 
          16:image.shape[-1]-16] = 0 # inpaint
    masked_image = image*mask
    print(masked_image.shape)
    masked_image = masked_image * 2 - 1
    model = initialize_model(eval_config, eval_config ["device"])
    ckpt = torch.load(os.path.join(eval_config["path_ckpt"]))
    ckpt = {key.replace('_orig_mod.', ""): value for key, value in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    print("Model weights loaded successfully.")
    model.eval()
    
    # sampler = GaussianDiffusionSampler(model=model, beta_1=eval_config["beta_1"], beta_T=eval_config["beta_T"], T=eval_config["T"], w=0.).to(eval_config ["device"])
    ddpm = GaussianDiffusionTrainer(model=model, 
                                    beta_1=eval_config["beta_1"], 
                                    beta_T=eval_config["beta_T"], 
                                    T=eval_config["T"],
                                    ).to(eval_config ["device"])
    scheduler = CustomScheduler.from_DDPMScheduler(ddpm)
    inpainted_image = repaint(masked_image.float(),
                              mask.float(),
                              model, 
                              scheduler,
                              j=50, 
                              r=20)
    
    
    # inpainted_image = sampler.inpaint(masked_image, mask, j=1, r=1, jumps_every=0)
    inpainted_image = torch.clamp(inpainted_image, -1, 1) # [-1,1]
    inpainted_image = (inpainted_image + 1 )/2  # [-1,1] --> [0,1]
    inpainted_image =  image * mask + (1-mask) * inpainted_image
    inpainted_image = inpainted_image > torch.mean(inpainted_image) # binary
    
    plt.imshow(inpainted_image[0, 0, :, :, inpainted_image.shape[-1]//2].cpu().numpy(), cmap="gray")
    plt.show()
    plt.close()
    plt.imshow(image[0, 0, :, :, image.shape[-1]//2].cpu().numpy(), cmap="gray")
    plt.show()
    plt.close()

