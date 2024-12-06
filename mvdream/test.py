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
import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera
import rembg
import os
import gradio as gr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from core.options import AllConfigs
from mvdream.pipeline_mvdream import MVDreamPipeline
import tyro

opt = tyro.cli(AllConfigs)


NUM_DDIM_STEPS = 50
SIZES = {
    0:4,
    1:2,
    2:1,
    3:1,
}
bg_remover = rembg.new_session()
pipe_image = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)

def process(input_image, prompt, prompt_neg='', input_elevation=0, input_num_steps=30, input_seed=42):

    # seed
    kiui.seed_everything(input_seed)

    os.makedirs(opt.workspace, exist_ok=True)

    
    input_image = np.array(input_image) # uint8
    # bg removal
    carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = carved_image[..., -1] > 0
    image = recenter(carved_image, mask, border_ratio=0.2)
    image = image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
    mv_image = pipe_image(prompt, image, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=5.0,  elevation=input_elevation)
    
    mv_image_grid = np.concatenate([
        np.concatenate([mv_image[1], mv_image[2]], axis=1),
        np.concatenate([mv_image[3], mv_image[0]], axis=1),
    ], axis=0)

    return mv_image_grid

class DragonModels():
    def __init__(self, pretrained_model_path='ashawkey/mvdream-sd2.1-diffusers'):
        self.editor = MVDreamPipeline.from_pretrained(
                pretrained_model_path, 
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )    

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


        latent = self.editor.image2latent(img_tensor)
        # def ddim_inv(self, latent, prompt, image=None, guidance_scale=7.0, num_frames=4, elevation=0, num_images_per_prompt=1, negative_prompt=""):
        ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt, negative_prompt=prompt_neg)
        return ddim_latents


_TITLE = '''LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation'''

_DESCRIPTION = '''
<div>
<a style="display:inline-block" href="https://me.kiui.moe/lgm/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/3DTopia/LGM"><img src='https://img.shields.io/github/stars/3DTopia/LGM?style=social'/></a>
</div>

* Input can be only text, only image, or both image and text. 
* If you find the output unsatisfying, try using different seeds!
'''

block = gr.Blocks(title=_TITLE).queue()
with block:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)
    
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            # input image
            input_image = gr.Image(label="image", type='pil')
            # input prompt
            input_text = gr.Textbox(label="prompt")
            # negative prompt
            input_neg_text = gr.Textbox(label="negative prompt", value='ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate')
            # elevation
            input_elevation = gr.Slider(label="elevation", minimum=-90, maximum=90, step=1, value=0)
            # inference steps
            input_num_steps = gr.Slider(label="inference steps", minimum=1, maximum=100, step=1, value=30)
            # random seed
            input_seed = gr.Slider(label="random seed", minimum=0, maximum=100000, step=1, value=0)
            # gen button
            button_gen = gr.Button("Generate")

        
        with gr.Column(scale=1):
            with gr.Tab("Multi-view Image"):
                output_image = gr.Image(interactive=False, show_label=False)

        button_gen.click(process, inputs=[input_image, input_text, input_neg_text, input_elevation, input_num_steps, input_seed], outputs=[output_image])
    
    gr.Examples(
        examples=[
            "data_test/anya_rgba.png",
            "data_test/bird_rgba.png",
            "data_test/catstatue_rgba.png",
        ],
        inputs=[input_image],
        outputs=[output_image],
        fn=lambda x: process(input_image=x, prompt=''),
        cache_examples=False,
        label='Image-to-3D Examples'
    )

    gr.Examples(
        examples=[
            "a motorbike",
            "a hamburger",
            "a furry red fox head",
        ],
        inputs=[input_text],
        outputs=[output_image],
        fn=lambda x: process(input_image=None, prompt=x),
        cache_examples=False,
        label='Text-to-3D Examples'
    )
    