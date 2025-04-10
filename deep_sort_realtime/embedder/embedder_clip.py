import os
import logging
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image

try:
    import clip
except ImportError:
    raise ImportError("CLIP no está instalado. Por favor instala con: pip install git+https://github.com/openai/CLIP.git")

logger = logging.getLogger(__name__)


def _batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx : min(ndx + bs, l)]


class Clip_Embedder(object):
    """
    Clip_Embedder loads a CLIP model of specified architecture, outputting a feature of size 1024 (or 768 for ViT-L/14).

    Params
    ------
    - model_name (optional, str) : CLIP model to use
    - model_wts_path (optional, str): Optional specification of path to CLIP model weights. Defaults to None and look for weights in `deep_sort_realtime/embedder/weights` or clip will download from internet into their own cache.
    - max_batch_size (optional, int) : max batch size for embedder, defaults to 16
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
    - gpu (optional, Bool) : boolean flag indicating if gpu is enabled or not, defaults to True
    - custom_input_resolution (optional, int) : custom input resolution for model, useful for ViT-L/14 which benefits from higher resolution
    """

    def __init__(
        self,
        model_name="ViT-B/32",
        model_wts_path=None,
        max_batch_size=16,
        bgr=True,
        gpu=True,
        custom_input_resolution=None,
    ):
        # Check if we're using a model that's supported by CLIP
        supported_models = [
            "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", 
            "ViT-B/32", "ViT-B/16", "ViT-L/14"
        ]
        
        self.is_vit_l_14 = model_name == "ViT-L/14"
        
        if not any(model_name == m or model_name.replace("-", "/") == m for m in supported_models):
            supported_str = ", ".join(f'"{m}"' for m in supported_models)
            raise ValueError(f"Modelo CLIP '{model_name}' no reconocido. Modelos soportados: {supported_str}")
            
        if model_wts_path is None:
            # Normalize model name for file paths
            model_name_normalized = model_name.replace("/", "-")
            
            # Check if we have local weights
            weights_path = (
                Path(__file__).parent.resolve() / "weights" / f"{model_name_normalized}.pt"
            )
            
            if weights_path.is_file():
                model_wts_path = str(weights_path)
                print(f"Usando pesos locales para CLIP: {model_wts_path}")
            else:
                # If not, we'll download from CLIP's servers
                model_wts_path = model_name
                print(f"Descargando pesos para CLIP modelo: {model_name}")

        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        
        try:
            self.model, self.img_preprocess = clip.load(model_wts_path, device=self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo CLIP {model_name}: {str(e)}")
        
        # Set custom input resolution if specified, especially useful for ViT-L/14
        self.custom_resolution = custom_input_resolution
        if self.is_vit_l_14 and not custom_input_resolution:
            # ViT-L/14 benefits from higher resolution inputs (336x336 suggested by OpenAI)
            self.custom_resolution = 336
            
        self.max_batch_size = max_batch_size
        self.bgr = bgr

        logger.info("Clip Embedder for Deep Sort initialised")
        logger.info(f"- gpu enabled: {gpu and torch.cuda.is_available()}")
        logger.info(f"- max batch size: {self.max_batch_size}")
        logger.info(f"- expects BGR: {self.bgr}")
        logger.info(f"- model name: {model_name}")
        if self.custom_resolution:
            logger.info(f"- custom input resolution: {self.custom_resolution}x{self.custom_resolution}")

        # Warmup with a small image
        zeros = np.zeros((100, 100, 3), dtype=np.uint8)
        self.predict([zeros])  # warmup
        
    def _preprocess_image(self, img_rgb):
        """Custom preprocessing for images, especially for ViT-L/14 which benefits from higher resolution"""
        if self.custom_resolution:
            # Resize to custom resolution while maintaining aspect ratio
            img_pil = Image.fromarray(img_rgb)
            
            # Resize with proper handling of aspect ratio
            img_pil = self._resize_with_padding(img_pil, self.custom_resolution)
            
            # Apply CLIP's standard preprocessing
            return self.img_preprocess(img_pil).to(self.device)
        else:
            # Use default CLIP preprocessing
            return self.img_preprocess(Image.fromarray(img_rgb)).to(self.device)
    
    def _resize_with_padding(self, pil_img, target_size):
        """Resize PIL image keeping aspect ratio and padding to square if needed"""
        width, height = pil_img.size
        
        # Determine scaling factor to fit within target_size
        scale = min(target_size / width, target_size / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize the image
        resized_img = pil_img.resize((new_width, new_height), Image.BICUBIC)
        
        # Create a black background
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        
        # Paste the resized image onto the center of the background
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        new_img.paste(resized_img, (paste_x, paste_y))
        
        return new_img

    def predict(self, np_images):
        """
        batch inference

        Params
        ------
        np_images : list of ndarray
            list of (H x W x C), bgr or rgb according to self.bgr

        Returns
        ------
        list of features (np.array with dim = 1024 for standard CLIP models, 768 for ViT-L/14)

        """
        if not np_images:
            return []

        # Convert BGR to RGB if necessary
        if self.bgr:
            np_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in np_images]

        # Apply custom preprocessing
        pil_images = [self._preprocess_image(rgb) for rgb in np_images]

        all_feats = []
        for this_batch in _batch(pil_images, bs=self.max_batch_size):
            batch = torch.stack(this_batch, 0)
            with torch.no_grad():
                feats = self.model.encode_image(batch)
                # Normalize features
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.extend(feats.cpu().data.numpy())
        return all_feats
