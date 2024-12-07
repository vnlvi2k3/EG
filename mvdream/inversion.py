from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import numpy as np
from mvdream.mv_unet import get_camera

class DDIMInversion:
    def __init__(self, unet, tokenizer, text_encoder, scheduler, guidance_scale, multiplier, actual_num_frames, camera, NUM_DDIM_STEPS=50):
        self.unet = unet
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.multiplier = multiplier
        self.actual_num_frames = actual_num_frames
        self.camera = camera
        self.guidance_scale = guidance_scale

        self.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.device = next(self.unet.parameters()).device
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, additional_inputs):
        latent_model_input = torch.cat([latents] * self.multiplier)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        unet_inputs = {
            'x': latent_model_input,
            'timesteps': torch.tensor([t] * self.actual_num_frames * self.multiplier, dtype=latent_model_input.dtype, device=self.device),
        }
        unet_inputs.update(additional_inputs)
        
        noise_pred = self.unet.forward(**unet_inputs)
        if self.multiplier > 1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return noise_pred

    @torch.no_grad()
    def init_prompt(self, prompt: str, emb_im=None):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_input.attention_mask.to(self.device)
        else:
            attention_mask = None
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device), attention_mask=attention_mask)[0]
        if emb_im is not None:
            self.text_embeddings = torch.cat([text_embeddings, emb_im],dim=1)
        else:
            self.text_embeddings = text_embeddings

        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, prompt_embeds, image_embeds):
        latent = latent.clone().detach()
        all_latent = [latent]
        print('DDIM Inversion:')
        prompt_embeds_neg, prompt_embeds_pos = prompt_embeds
        additional_inputs = {
            'context': torch.cat([prompt_embeds_neg] * self.actual_num_frames + [prompt_embeds_pos] * self.actual_num_frames),
            'num_frames':  self.actual_num_frames,
            'camera': torch.cat([self.camera] * self.multiplier),
        }
        if image_embeds is not None:
            additional_inputs['ip'] = image_embeds[0]
            additional_inputs['ip_img'] = image_embeds[1]

        for i in tqdm(range(self.NUM_DDIM_STEPS)):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, additional_inputs)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)

        return all_latent
    
    def invert(self, ddim_latents, prompt_embeds, image_embeds):
        # self.init_prompt(prompt, emb_im=emb_im)
        ddim_latents = self.ddim_loop(ddim_latents, prompt_embeds, image_embeds)
        return ddim_latents