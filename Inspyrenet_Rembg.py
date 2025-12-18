from PIL import Image
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm
import os

# Try to import folder_paths for ComfyUI integration
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("Warning: folder_paths module not found. Custom model directory support disabled.")


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def get_model_path(mode="base"):
    """
    Get the model checkpoint path, checking ComfyUI models directory first.

    Args:
        mode (str): Model mode - 'base', 'fast', or 'base-nightly'

    Returns:
        str or None: Path to model checkpoint if found in ComfyUI models directory, None otherwise

    Model download links:
        base:         https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_base.pth
        fast:         https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_fast.pth
        base-nightly: https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_base_nightly.pth
    """
    if not COMFYUI_AVAILABLE:
        return None

    # Model filename mapping based on transparent_background's config.yaml
    model_filenames = {
        "base": "ckpt_base.pth",
        "fast": "ckpt_fast.pth",
        "base-nightly": "ckpt_base_nightly.pth"
    }

    model_filename = model_filenames.get(mode, "ckpt_base.pth")

    # Try to get ComfyUI base directory
    try:
        # Get the models directory from folder_paths
        if hasattr(folder_paths, 'models_dir'):
            comfy_models_dir = folder_paths.models_dir
        else:
            # Fallback: try to infer from folder_paths location
            import folder_paths as fp
            comfy_base = os.path.dirname(os.path.dirname(os.path.abspath(fp.__file__)))
            comfy_models_dir = os.path.join(comfy_base, "models")

        # Check in inspyrenet_models subdirectory
        inspyrenet_models_dir = os.path.join(comfy_models_dir, "inspyrenet_models")
        model_path = os.path.join(inspyrenet_models_dir, model_filename)

        if os.path.exists(model_path):
            print(f"[Inspyrenet-Rembg] Using model from ComfyUI models directory: {model_path}")
            return model_path
        else:
            print(f"[Inspyrenet-Rembg] Model not found in {inspyrenet_models_dir}, will use default download location")
            return None

    except Exception as e:
        print(f"[Inspyrenet-Rembg] Error checking ComfyUI models directory: {e}")
        return None

class InspyrenetRembg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, torchscript_jit):
        # Check for custom model path in ComfyUI models directory
        custom_ckpt = get_model_path(mode="base")

        if (torchscript_jit == "default"):
            remover = Remover(ckpt=custom_ckpt) if custom_ckpt else Remover()
        else:
            remover = Remover(jit=True, ckpt=custom_ckpt) if custom_ckpt else Remover(jit=True)

        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type='rgba')
            out =  pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        return (img_stack, mask)
        
class InspyrenetRembgAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, torchscript_jit, threshold):
        # Check for custom model path in ComfyUI models directory
        custom_ckpt = get_model_path(mode="base")

        if (torchscript_jit == "default"):
            remover = Remover(ckpt=custom_ckpt) if custom_ckpt else Remover()
        else:
            remover = Remover(jit=True, ckpt=custom_ckpt) if custom_ckpt else Remover(jit=True)

        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type='rgba', threshold=threshold)
            out =  pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        return (img_stack, mask)