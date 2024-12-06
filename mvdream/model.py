from pipeline_mvdream import MVDreamPipeline

from core.utils import process_drag, resize_numpy_image
import torch
import cv2
from pytorch_lightning import seed_everything
from PIL import Image
from torchvision.transforms import PILToTensor
import numpy as np
import torch.nn.functional as F
import dlib

NUM_DDIM_STEPS = 50
SIZES = {
    0:4,
    1:2,
    2:1,
    3:1,
}

class DragonModels():
    def __init__(self, pretrained_model_path='ashawkey/mvdream-sd2.1-diffusers'):
        self.ip_scale = 0.1
        self.precision = torch.float16
        self.editor = MVDreamPipeline.from_pretrained(
                pretrained_model_path, 
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        self.up_ft_index = [1,2] 
        self.up_scale = 2        
        self.device = 'cuda'     

    def run_drag(self, original_image, mask, prompt, prompt_neg, w_edit, w_content, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img = original_image
        img, input_scale = resize_numpy_image(img, max_resolution*max_resolution)
        h, w = img.shape[1], img.shape[0] 
        img = Image.fromarray(img)
        img_prompt = img.resize((256, 256))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)
        mask = np.repeat(mask[:,:,None], 3, 2) if len(mask.shape)==2 else mask

        emb_im, emb_im_uncond = self.editor.get_image_embeds(img_prompt)
        #tạm thời ko scale, ko cho đổi ip_scale the slider 
        # if ip_scale is not None and ip_scale != self.ip_scale:
        #     self.ip_scale = ip_scale
        #     self.editor.load_adapter(self.editor.ip_id, self.ip_scale)

        latent = self.editor.image2latent(img_tensor)
        # def ddim_inv(self, latent, prompt, image=None, guidance_scale=7.0, num_frames=4, elevation=0, num_images_per_prompt=1, negative_prompt=""):
        ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt, negative_prompt=prompt_neg)
        latent_in = ddim_latents[-1].squeeze(2)

        x=[]
        y=[]
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx%2 == 0:
                y.append(point[1]*input_scale)
                x.append(point[0]*input_scale)
            else:
                y_cur.append(point[1]*input_scale)
                x_cur.append(point[0]*input_scale)
        
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        edit_kwargs = process_drag(
            latent_in = latent_in,
            path_mask=mask, 
            h=h, 
            w=w, 
            x=x,
            y=y,
            x_cur=x_cur,
            y_cur=y_cur,
            scale=scale, 
            input_scale=input_scale, 
            up_scale=self.up_scale, 
            up_ft_index=self.up_ft_index, 
            w_edit=w_edit, 
            w_content=w_content, 
            w_inpaint=w_inpaint,  
            precision=self.precision, 
        )
        latent_in = edit_kwargs.pop('latent_in')
        latent_rec = self.editor.edit(
            mode = 'drag',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in, 
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            energy_scale=energy_scale, 
            latent_noise_ref = ddim_latents,
            SDE_strength=SDE_strength, 
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]


