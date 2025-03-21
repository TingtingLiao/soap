from typing import Any, Callable, Dict, List, Optional, Tuple, Union 
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.schedulers import KarrasDiffusionSchedulers 
from diffusers import AutoencoderKL, StableDiffusionImageVariationPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker, StableDiffusionPipelineOutput 


class Unique3dDiffusionPipeline(StableDiffusionImageVariationPipeline):       
    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNet2DConditionModel, 
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        class_labels: Optional[torch.Tensor] = None,  
    ):
        super().__init__(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker
        )   
        self.class_labels = class_labels   

    def encode_latents(self, image: Image.Image, height, width):
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        elif isinstance(image, List):
            image = [img.convert("RGB") for img in image]
        
        image = self.image_processor.preprocess(image, height=height, width=width)
        image = image.to(self._execution_device, dtype=self.dtype) 
        latents = self.vae.encode(image).latent_dist.mode() * self.vae.config.scaling_factor
        return latents
    
    def decode_latents(self, latents): 
        latents = latents / self.vae.config.scaling_factor  
        imgs = self.vae.decode(latents).sample 
        return imgs
    
    def encode_image(self, image, num_images_per_prompt):  
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=self._execution_device, dtype=self.dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        negative_prompt_embeds = torch.zeros_like(image_embeddings) 
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings

    def train_step(self):
        # TODO: Implement sds optimization 
        pass 
    
    @torch.no_grad()
    def produce_latent(
        self, latents, cond_latents, embeddings, 
        num_inference_steps=50,  
        init_step=0, 
        guidance_scale=1.5, 
        extra_step_kwargs=dict(),
    ):
        timesteps = self.scheduler.timesteps  
        with self.progress_bar(total=num_inference_steps) as pbar:  
            for i, t in enumerate(timesteps[init_step:]): 
                latent_model_input = torch.cat([latents] * 2)  
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  

                noise_pred = self.unet( 
                    latent_model_input, 
                    t, 
                    embeddings, 
                    condition_latens=cond_latents,
                    class_labels=self.class_labels,  
                ).sample 
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    pbar.update()  
                    
        return latents
    
    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, List[Image.Image], torch.FloatTensor], 
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        height_cond: Optional[int] = 512,
        width_cond: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, 
        output_type = "pil",
        strength: float = 0.0,
    ):  
        self.check_inputs(image, height, width, 1)
        
        batch = len(image) if isinstance(image, List) else 1 
        
        # prepare inputs
        device = self._execution_device 
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor 
        self.scheduler.set_timesteps(num_inference_steps, device=device) 
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        init_step = min(int(len(self.scheduler.timesteps)*strength), len(self.scheduler.timesteps)-1)  
        
        # embeddings
        image_embeddings = self.encode_image(image, num_images_per_prompt)

        # condition latents 
        cond_latents = self.encode_latents(image, height_cond, width_cond)
        cond_latents = torch.cat([torch.zeros_like(cond_latents), cond_latents])
        if num_images_per_prompt > 1: 
            cond_latents = torch.stack([cond_latents] * num_images_per_prompt, 1).reshape(-1, *cond_latents.shape[1:]) 
        
        # latents  
        latents = self.prepare_latents(
            batch * num_images_per_prompt,
            self.unet.config.out_channels,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator, 
        )
        
        if self.class_labels is not None:
            self.class_labels = self.class_labels.repeat(len(image_embeddings)// len(self.class_labels)).to(device)
        
        # optimize latents
        latents = self.produce_latent(
            latents, cond_latents, image_embeddings, num_inference_steps, init_step, guidance_scale, extra_step_kwargs
            )  
        
        # decode 
        image = self.decode_latents(latents)
        
        if output_type == "torch":
            image = ((image.permute(0, 2, 3, 1) + 1) * 0.5).clamp(0, 1) # [B, 3, H, W]
        else:
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=[True]*len(image))
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    @torch.no_grad()
    def refine(
        self,
        image: Union[Image.Image, List[Image.Image], torch.FloatTensor],
        emb_image: Optional[Union[Image.Image, List[Image.Image], torch.FloatTensor]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        height_cond: Optional[int] = 512,
        width_cond: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, 
        output_type: Optional[str] = "pil", 
        strength: float = 0.0,
    ): 
        # prepare inputs 
        device = self._execution_device
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor 
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta) 
        emb_image = image if emb_image is None else emb_image
        self.scheduler.set_timesteps(num_inference_steps)
        init_step = min(int(len(self.scheduler.timesteps)*strength), len(self.scheduler.timesteps)-1)  
        
        # embeddings 
        image_embeddings = self.encode_image(emb_image, num_images_per_prompt)

        # condition latents
        cond_latents = self.encode_latents(emb_image, height_cond, width_cond) 
        cond_latents = torch.cat([torch.zeros_like(cond_latents), cond_latents])
        if num_images_per_prompt > 1:
            cond_latents = torch.stack([cond_latents] * num_images_per_prompt, 1).reshape(-1, *cond_latents.shape[1:])
        
        # latents  
        if strength == 0:  
            latents = self.prepare_latents(
                4,
                self.unet.config.out_channels,
                height,
                width,
                image_embeddings.dtype,
                device,
                generator, 
            )
        else: 
            latents = torch.cat([
                self.encode_latents(img, height, width) for img in image
            ]) 
            noise = self.prepare_latents(
                4,
                self.unet.config.out_channels,
                height,
                width,
                image_embeddings.dtype,
                device,
                generator, 
            ) 
        
            latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[init_step][None])
            
        if self.class_labels is not None:
            self.class_labels = self.class_labels.repeat(len(image_embeddings)// len(self.class_labels)).to(device)
            
        # optimize latents
        latents = self.produce_latent(
            latents, cond_latents, image_embeddings, num_inference_steps, init_step, guidance_scale, extra_step_kwargs
            )
        
        # decode 
        image = self.decode_latents(latents) 
        
        if output_type == "pil":
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=[True]*len(image))
        else:
            image = ((image + 1) / 2).clamp(0, 1) # [B, 3, H, W]

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
