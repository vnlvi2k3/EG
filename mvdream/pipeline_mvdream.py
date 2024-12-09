import torch
import torch.nn.functional as F
import inspect
import numpy as np
from typing import Callable, List, Optional, Union
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
import gc
from tqdm import tqdm
import torch.nn as nn
from mvdream.inversion import DDIMInversion
from PIL import Image
import copy

from mvdream.mv_unet import MultiViewUNetModel, MyMultiViewUNetModel, get_camera
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MVDreamPipeline(DiffusionPipeline):

    _optional_components = ["feature_extractor", "image_encoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: MultiViewUNetModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        scheduler: DDIMScheduler,
        # imagedream variant
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModel,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:  # type: ignore
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "  # type: ignore
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:  # type: ignore
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        # init_signature = inspect.signature(unet.__class__.__init__)
        # params = list(init_signature.parameters.values())[1:]
        # params = [param for param in params if hasattr(unet, param.name)]
        # estimator = MyMultiViewUNetModel(**{param.name: getattr(unet, param.name) for param in params})
        # estimator.load_state_dict(unet.state_dict())
        estimator = MyMultiViewUNetModel.from_config('pretrained/config.json').half()
        estimator.load_state_dict(unet.state_dict(), strict=True)


        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            estimator=estimator,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher"
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                "`enable_model_offload` requires `accelerate v0.17.0` or higher."
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook
            )

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance: bool,
        negative_prompt=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` should be either a string or a list of strings, but got {type(prompt)}."
            )

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
            
        image = self.feature_extractor(image, return_tensors="pt").pixel_values
        image = image.to(device=device, dtype=dtype)
        
        image_embeds = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return torch.zeros_like(image_embeds), image_embeds

    def encode_image_latents(self, image, device, num_images_per_prompt):
        
        dtype = next(self.image_encoder.parameters()).dtype

        image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).to(device=device) # [1, 3, H, W]
        image = 2 * image - 1
        image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
        image = image.to(dtype=dtype)

        posterior = self.vae.encode(image).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor # [B, C, H, W]
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)

        return torch.zeros_like(latents), latents
  
    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            dtype = next(self.image_encoder.parameters()).dtype
            # latents = self.vae.encode(image)['latent_dist'].mean
            # latents = latents *  self.vae.config.scaling_factor
            image = torch.tensor(image).permute(0, 3, 1, 2).to('cuda') 
            image = image.to(dtype)
            image = 2 * image - 1
            latents = self.vae.encode(image)['latent_dist'].mean
            latents = latents *  self.vae.config.scaling_factor
        return latents
    
    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        dtype = next(self.image_encoder.parameters()).dtype
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to('cuda', dtype=dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_embed(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2].detach()
        uncond_image_prompt_embeds = self.image_embed(uncond_clip_image_embeds).detach()
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def ddim_inv(self, latent, prompt, image=None, guidance_scale=7.0, num_frames=4, elevation=0, num_images_per_prompt=1, negative_prompt=""):
        do_classifier_free_guidance = guidance_scale > 1.0
        multiplier = 2 if do_classifier_free_guidance else 1

        actual_num_frames = num_frames if image is None else num_frames + 1
        if image is not None:
            camera = get_camera(num_frames, elevation=elevation, extra_view=True).to(dtype=latent.dtype, device=self.device)
        else:
            camera = get_camera(num_frames, elevation=elevation, extra_view=False).to(dtype=latent.dtype, device=self.device)
        camera = camera.repeat_interleave(num_images_per_prompt, dim=0)

        with torch.no_grad():
            _prompt_embeds = self._encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )  
        prompt_embeds_neg, prompt_embeds_pos = _prompt_embeds.chunk(2)
        prompt_embeds=[prompt_embeds_neg, prompt_embeds_pos]

        image_embeds = None
        if image is not None:
            with torch.no_grad():
                image_embeds_neg, image_embeds_pos = self.encode_image(image, self.device, num_images_per_prompt)
                image_latents_neg, image_latents_pos = self.encode_image_latents(image, self.device, num_images_per_prompt)
            ip = torch.cat([image_embeds_neg] * actual_num_frames + [image_embeds_pos] * actual_num_frames)
            ip_img = torch.cat([image_latents_neg] + [image_latents_pos]) # no repeat
            image_embeds=[ip, ip_img]
        ddim_inv = DDIMInversion(self.unet, self.tokenizer, self.text_encoder, self.scheduler, guidance_scale, multiplier, actual_num_frames, camera)
        ddim_latents = ddim_inv.invert(ddim_latents=latent, prompt_embeds=prompt_embeds, image_embeds=image_embeds)
        return ddim_latents
    
    def edit(
        self,
        prompt:  List[str],
        mode,
        image,
        edit_kwargs,
        num_inference_steps: int = 50,
        elevation: float = 0,
        guidance_scale: Optional[float] = 7.0,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        latent: Optional[torch.FloatTensor] = None,
        start_time=50,
        energy_scale = 0,
        SDE_strength = 0.4,
        SDE_strength_un = 0,
        latent_noise_ref = None,
        num_frames: int = 4,
        alg='D+'
    ):
        self.unet = self.unet.to(device='cuda')
        self.vae = self.vae.to(device='cuda')
        self.text_encoder = self.text_encoder.to(device='cuda')
        self.estimator = self.estimator.to(device='cuda')
        print('Start Editing:')
        self.alg=alg
        do_classifier_free_guidance = guidance_scale > 1.0
        multiplier = 2 if do_classifier_free_guidance else 1

        if image is not None:
            assert isinstance(image, np.ndarray) and image.dtype == np.float32
            self.image_encoder = self.image_encoder.to(device=self.device)
            image_embeds_neg, image_embeds_pos = self.encode_image(image, self.device, num_images_per_prompt)
            image_latents_neg, image_latents_pos = self.encode_image_latents(image, self.device, num_images_per_prompt)
        
        _prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )  # type: ignore
        prompt_embeds_neg, prompt_embeds_pos = _prompt_embeds.chunk(2)
        actual_num_frames = num_frames if image is None else num_frames + 1

        # generate source text embedding
        # text_input = self.tokenizer(
        #     [prompt],
        #     padding="max_length",
        #     max_length=self.tokenizer.model_max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        # text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        # max_length = text_input.input_ids.shape[-1]
        # uncond_input = self.tokenizer(
        #         [""], padding="max_length", max_length=max_length, return_tensors="pt"
        #     )
        # uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        # # image prompt
        # if emb_im is not None and emb_im_uncond is not None:
        #     uncond_embeddings = torch.cat([uncond_embeddings, emb_im_uncond],dim=1)
        #     text_embeddings_org = text_embeddings
        #     text_embeddings = torch.cat([text_embeddings, emb_im],dim=1)
        #     context = torch.cat([uncond_embeddings.expand(*text_embeddings.shape), text_embeddings])

        self.scheduler.set_timesteps(num_inference_steps) 
        dict_mask = edit_kwargs['dict_mask'] if 'dict_mask' in edit_kwargs else None

        

        if image is not None:
            camera = get_camera(num_frames, elevation=elevation, extra_view=True).to(dtype=latent.dtype, device=self.device)
        else:
            camera = get_camera(num_frames, elevation=elevation, extra_view=False).to(dtype=latent.dtype, device=self.device)
        camera = camera.repeat_interleave(num_images_per_prompt, dim=0) 
        context = torch.cat([prompt_embeds_neg] * actual_num_frames + [prompt_embeds_pos] * actual_num_frames)
        cam = torch.cat([camera] * multiplier)
        additional_inputs = {
            'context': context,
            'num_frames': actual_num_frames,
            'camera': cam,
        }
        if image is not None:
            ip = torch.cat([image_embeds_neg] * actual_num_frames + [image_embeds_pos] * actual_num_frames)
            ip_image = torch.cat([image_latents_neg] + [image_latents_pos])
            additional_inputs['ip'] = ip
            additional_inputs['ip_img'] = ip_image
        for i, t in enumerate(tqdm(self.scheduler.timesteps[-start_time:])):
            latent_model_input = torch.cat([latent] * multiplier)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
            next_timestep = max(next_timestep, 0)
            if energy_scale==0 or alg=='D':
                repeat=1
            elif 20<i<30 and i%2==0 : 
                repeat = 3
            else:
                repeat = 1
            stack = []
            for ri in range(repeat):
                # latent_in = torch.cat([latent.unsqueeze(2)] * 2)
                tim = torch.tensor([t] * actual_num_frames * multiplier, dtype=latent_model_input.dtype, device='cuda')
                unet_inputs = {
                    'x': latent_model_input,
                    'timesteps': tim,
                }
                unet_inputs.update(additional_inputs)
                with torch.no_grad():
                    # noise_pred = self.unet(latent_in, t, encoder_hidden_states=context, mask=dict_mask, save_kv=False, mode=mode, iter_cur=i)["sample"].squeeze(2)
                    noise_pred = self.unet.forward(**unet_inputs)
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

                if energy_scale!=0 and i<30 and (alg=='D' or i%2==0 or i<10):
                    # editing guidance
                    noise_pred_org = noise_pred
                    if mode == 'drag':
                        latent_noise_ref_input = latent_noise_ref[-(i+1)]
                        latent_noise_ref_input = torch.cat([latent_noise_ref_input] * multiplier)
                        latent_noise_ref_input = self.scheduler.scale_model_input(latent_noise_ref_input, t)

                        guidance = self.guidance_drag(latent=latent_model_input, latent_noise_ref=latent_noise_ref_input, t=tim, energy_scale=energy_scale, additional_inputs= additional_inputs, **edit_kwargs)
                    _, guidance = guidance.chunk(2)
                    noise_pred = noise_pred + guidance
                else:
                    noise_pred_org=None
                # zt->zt-1
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latent - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

                if 10<i<20:
                    eta, eta_rd = SDE_strength_un, SDE_strength
                else:
                    eta, eta_rd = 0., 0.
                
                variance = self.scheduler._get_variance(t, prev_timestep)
                std_dev_t = eta * variance ** (0.5)
                std_dev_t_rd = eta_rd * variance ** (0.5)
                if noise_pred_org is not None:
                    pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred_org
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred_org
                else:
                    pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

                latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                latent_prev_rd = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_rd

                # Regional SDE
                if (eta_rd > 0 or eta>0) and alg=='D+':
                    variance_noise = torch.randn_like(latent_prev)
                    variance_rd = std_dev_t_rd * variance_noise
                    variance = std_dev_t * variance_noise
                    
                    if mode == 'drag':
                        mask = F.interpolate(edit_kwargs["mask_x0"].unsqueeze(1), (latent_prev[-1].shape[-2], latent_prev[-1].shape[-1]))
                        mask = (mask>0).to(dtype=latent.dtype)
                if repeat>1:
                    with torch.no_grad():
                        alpha_prod_t = self.scheduler.alphas_cumprod[next_timestep]
                        alpha_prod_t_next = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t

                        next_tim = torch.tensor([next_timestep] * actual_num_frames * multiplier, dtype=latent_model_input.dtype, device=self.device)
                        latent_prev_input = torch.cat([latent_prev] * multiplier)
                        latent_prev_input = self.scheduler.scale_model_input(latent_prev_input, t)
                        unet_inputs = {
                            'x': latent_prev_input,
                            'timesteps': next_tim,
                        }
                        unet_inputs.update(additional_inputs)
                        model_output = self.unet.forward(**unet_inputs)
                        _, model_output = model_output.chunk(2)
                        next_original_sample = (latent_prev - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
                        latent = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
            
            latent = latent_prev
            
        return latent

    def guidance_drag(
        self, 
        mask_x0,
        mask_cur, 
        mask_tar, 
        mask_other, 
        latent, 
        latent_noise_ref, 
        up_ft_index, 
        up_scale, 
        energy_scale,
        w_edit,
        w_inpaint,
        w_content,
        t,
        additional_inputs,
        dict_mask = None,
    ):
        mask_x0 = torch.cat([mask_x0] * 2)
        mask_other = torch.cat([mask_other] * 2)
        cos = nn.CosineSimilarity(dim=1)
        ref_inputs = {
            'x': latent_noise_ref,
            'timesteps': t,
            'up_ft_indices': up_ft_index,
        }
        ref_inputs.update(additional_inputs)
        with torch.no_grad():
            ref_inputs['ip_img'] = ref_inputs['ip_img'].detach().clone().requires_grad_(False)
            ref_inputs['ip'] = ref_inputs['ip'].detach().clone().requires_grad_(False)
            up_ft_tar = self.estimator.forward(**ref_inputs)['up_ft']
            for f_id in range(len(up_ft_tar)):
                up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (up_ft_tar[-1].shape[-2]*up_scale, up_ft_tar[-1].shape[-1]*up_scale))

        latent = latent.detach().requires_grad_(True)
        cur_inputs = {
            'x': latent,
            'timesteps': t,
            'up_ft_indices': up_ft_index,
        }
        cur_inputs.update(additional_inputs)
        up_ft_cur = self.estimator.forward(**cur_inputs)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))

        # moving loss
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar):
                mask_cur_i = torch.cat([mask_cur_i] * 2).unsqueeze(1).bool()
                mask_tar_i = torch.cat([mask_tar_i] * 2).unsqueeze(1).bool()
                up_ft_cur_vec = up_ft_cur[f_id][mask_cur_i.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
                up_ft_tar_vec = up_ft_tar[f_id][mask_tar_i.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
                sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
                loss_edit = loss_edit + w_edit/(1+4*sim.mean())

                mask_overlap = ((mask_cur_i.float()+mask_tar_i.float())>1.5).float()
                mask_non_overlap = (mask_tar_i.float()-mask_overlap)>0.5
                up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
                up_ft_tar_non_overlap = up_ft_tar[f_id][mask_non_overlap.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
                sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
                loss_edit = loss_edit + w_inpaint*sim_non_overlap.mean()
        # consistency loss
        loss_con = 0
        for f_id in range(len(up_ft_tar)):
            sim_other = (cos(up_ft_tar[f_id], up_ft_cur[f_id])[mask_other.squeeze(1)]+1.)/2.
            loss_con = loss_con+w_content/(1+4*sim_other.mean())
        loss_edit = loss_edit/len(up_ft_cur)/len(mask_cur)
        loss_con = loss_con/len(up_ft_cur)

        
        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        mask = F.interpolate(mask_x0.unsqueeze(1), (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]))
        mask = (mask>0).to(dtype=latent.dtype)
        guidance = cond_grad_edit.detach()*4e-2*mask + cond_grad_con.detach()*4e-2*(1-mask)
        self.estimator.zero_grad()

        return guidance
            

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        image: Optional[np.ndarray] = None,
        height: int = 256,
        width: int = 256,
        elevation: float = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "numpy", # pil, numpy, latents
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        num_frames: int = 4,
        device=torch.device("cuda:0"),
    ):
        self.unet = self.unet.to(device=device)
        self.vae = self.vae.to(device=device)
        self.text_encoder = self.text_encoder.to(device=device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # imagedream variant
        if image is not None:
            assert isinstance(image, np.ndarray) and image.dtype == np.float32
            self.image_encoder = self.image_encoder.to(device=device)
            image_embeds_neg, image_embeds_pos = self.encode_image(image, device, num_images_per_prompt)
            image_latents_neg, image_latents_pos = self.encode_image_latents(image, device, num_images_per_prompt)
            
        _prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )  # type: ignore
        prompt_embeds_neg, prompt_embeds_pos = _prompt_embeds.chunk(2)

        # Prepare latent variables
        actual_num_frames = num_frames if image is None else num_frames + 1
        latents: torch.Tensor = self.prepare_latents(
            actual_num_frames * num_images_per_prompt,
            4,
            height,
            width,
            prompt_embeds_pos.dtype,
            device,
            generator,
            None,
        )

        if image is not None:
            camera = get_camera(num_frames, elevation=elevation, extra_view=True).to(dtype=latents.dtype, device=device)
        else:
            camera = get_camera(num_frames, elevation=elevation, extra_view=False).to(dtype=latents.dtype, device=device)
        camera = camera.repeat_interleave(num_images_per_prompt, dim=0)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                multiplier = 2 if do_classifier_free_guidance else 1
                latent_model_input = torch.cat([latents] * multiplier)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                unet_inputs = {
                    'x': latent_model_input,
                    'timesteps': torch.tensor([t] * actual_num_frames * multiplier, dtype=latent_model_input.dtype, device=device),
                    'context': torch.cat([prompt_embeds_neg] * actual_num_frames + [prompt_embeds_pos] * actual_num_frames),
                    'num_frames': actual_num_frames,
                    'camera': torch.cat([camera] * multiplier),
                }

                if image is not None:
                    unet_inputs['ip'] = torch.cat([image_embeds_neg] * actual_num_frames + [image_embeds_pos] * actual_num_frames)
                    unet_inputs['ip_img'] = torch.cat([image_latents_neg] + [image_latents_pos]) # no repeat
                
                # predict the noise residual
                noise_pred = self.unet.forward(**unet_inputs)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents: torch.Tensor = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)  # type: ignore

        # Post-processing
        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else: # numpy
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return image