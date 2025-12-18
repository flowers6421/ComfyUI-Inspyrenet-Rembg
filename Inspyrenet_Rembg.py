import time
import sys

# ============================================================================
# DEBUG LOGGING - Track initialization and performance
# ============================================================================
DEBUG_PREFIX = "[Inspyrenet-Rembg-DEBUG]"
_import_start_time = time.time()

def debug_log(message, start_time=None):
    """Print debug message with optional elapsed time."""
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"{DEBUG_PREFIX} {message} (took {elapsed:.3f}s)", flush=True)
    else:
        print(f"{DEBUG_PREFIX} {message}", flush=True)

debug_log("Starting imports...")

# ============================================================================
# IMPORTS WITH TIMING
# ============================================================================
_t = time.time()
from PIL import Image
debug_log("Imported PIL.Image", _t)

_t = time.time()
import torch
debug_log(f"Imported torch (version: {torch.__version__}, CUDA available: {torch.cuda.is_available()})", _t)

if torch.cuda.is_available():
    debug_log(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    debug_log(f"  CUDA version: {torch.version.cuda}")

_t = time.time()
import numpy as np
debug_log("Imported numpy", _t)

_t = time.time()
from tqdm import tqdm
debug_log("Imported tqdm", _t)

_t = time.time()
import os
debug_log("Imported os", _t)

# Import transparent_background with detailed timing
debug_log("Importing transparent_background (this may trigger dependency loading)...")

_t = time.time()
import albumentations
debug_log("Imported albumentations", _t)

_t = time.time()
import timm
debug_log(f"Imported timm (version: {timm.__version__})", _t)

_t = time.time()
try:
    from transparent_background import Remover
    debug_log("Imported transparent_background.Remover", _t)
except Exception as e:
    debug_log(f"ERROR importing transparent_background: {e}", _t)
    raise

# Log total import time
debug_log(f"All imports completed", _import_start_time)

# Try to import folder_paths for ComfyUI integration
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
    debug_log("folder_paths module available (ComfyUI environment detected)")
except ImportError:
    COMFYUI_AVAILABLE = False
    debug_log("folder_paths module not found (not in ComfyUI environment)")

# Track if this is the first Remover instantiation
_first_remover_init = True
_first_process_call = True


# ============================================================================
# FAST REMOVER - Bypasses slow transparent_background.Remover initialization
# The standard Remover takes 68s due to MD5 verification or re-downloading.
# This version loads the model directly in ~1.5s.
# ============================================================================
class FastRemover:
    """
    Fast replacement for transparent_background.Remover that skips slow initialization.
    Loads model directly without MD5 verification or checkpoint downloading.
    """

    def __init__(self, mode="base", jit=False, device=None, ckpt=None):
        import shutil
        from transparent_background.InSPyReNet import InSPyReNet_SwinB
        from transparent_background.utils import load_config
        import albumentations as A
        import albumentations.pytorch as AP

        total_start = time.time()
        debug_log("=" * 60)
        debug_log("FAST REMOVER INITIALIZATION (bypassing slow Remover)")
        debug_log(f"  mode={mode}, jit={jit}, device={device}, ckpt={ckpt}")

        self.mode = mode
        self.jit = jit

        # Step 1: Load config
        _t = time.time()
        cfg_path = os.environ.get('TRANSPARENT_BACKGROUND_FILE_PATH', os.path.abspath(os.path.expanduser('~')))
        home_dir = os.path.join(cfg_path, ".transparent-background")
        os.makedirs(home_dir, exist_ok=True)

        repopath = os.path.dirname(os.path.abspath(__import__('transparent_background').__file__))
        if not os.path.isfile(os.path.join(home_dir, "config.yaml")):
            shutil.copy(os.path.join(repopath, "config.yaml"), os.path.join(home_dir, "config.yaml"))
        self.meta = load_config(os.path.join(home_dir, "config.yaml"))[mode]
        debug_log("Config loaded", _t)

        # Step 2: Determine device
        _t = time.time()
        if device is not None:
            self.device = device
        else:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda:0"
        debug_log(f"Device: {self.device}", _t)

        # Step 3: Resolve checkpoint path (NO MD5 verification - that's what makes standard Remover slow!)
        _t = time.time()
        if ckpt is not None:
            ckpt_path = os.path.abspath(ckpt)
        else:
            ckpt_path = os.path.join(home_dir, self.meta.ckpt_name)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Please download it first.")

        ckpt_size = os.path.getsize(ckpt_path) / (1024 * 1024)
        debug_log(f"Checkpoint: {ckpt_path} ({ckpt_size:.2f} MB)", _t)

        # Step 4: Create model
        _t = time.time()
        self.model = InSPyReNet_SwinB(depth=64, pretrained=False, threshold=None, **self.meta)
        self.model.eval()
        debug_log("Model created", _t)

        # Step 5: Load weights
        _t = time.time()
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        debug_log("Weights loaded", _t)

        # Step 6: Move to device
        _t = time.time()
        self.model = self.model.to(self.device)
        if 'cuda' in self.device:
            torch.cuda.synchronize()
        debug_log("Model on GPU", _t)

        # Step 7: JIT compile if requested
        if jit:
            _t = time.time()
            self.model = torch.jit.trace(
                self.model,
                torch.rand(1, 3, *self.meta.base_size).to(self.device),
                strict=False
            )
            debug_log("JIT compiled", _t)

        # Setup transforms (same as original Remover)
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AP.ToTensorV2()
        ])

        debug_log(f"FAST REMOVER READY", total_start)
        debug_log("=" * 60)

    def process(self, img, type='rgba', threshold=None):
        """Process image for background removal. Matches Remover.process() API."""
        import torch.nn.functional as F
        from PIL import Image

        # Get original size
        if isinstance(img, Image.Image):
            orig_size = img.size  # (W, H)
            img_np = np.array(img)
        else:
            img_np = img
            orig_size = (img_np.shape[1], img_np.shape[0])

        # Resize to model input size
        base_size = self.meta.base_size  # [H, W]
        img_resized = np.array(Image.fromarray(img_np).resize((base_size[1], base_size[0]), Image.BILINEAR))

        # Transform
        transformed = self.transform(image=img_resized)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            pred = self.model(img_tensor)

        # Get mask from prediction
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]  # Get final prediction

        # Resize mask back to original size
        mask = F.interpolate(pred, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
        mask = mask.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        # Apply threshold if specified
        if threshold is not None:
            mask = np.where(mask > int(threshold * 255), 255, 0).astype(np.uint8)

        # Create output based on type
        if type == 'map':
            return Image.fromarray(mask)
        elif type == 'rgba':
            img_pil = Image.fromarray(img_np).convert('RGBA')
            mask_pil = Image.fromarray(mask)
            img_pil.putalpha(mask_pil)
            return img_pil
        elif type == 'green':
            img_pil = Image.fromarray(img_np).convert('RGBA')
            mask_pil = Image.fromarray(mask)
            green_bg = Image.new('RGBA', img_pil.size, (0, 255, 0, 255))
            img_pil.putalpha(mask_pil)
            result = Image.alpha_composite(green_bg, img_pil)
            return result.convert('RGB')
        elif type == 'white':
            img_pil = Image.fromarray(img_np).convert('RGBA')
            mask_pil = Image.fromarray(mask)
            white_bg = Image.new('RGBA', img_pil.size, (255, 255, 255, 255))
            img_pil.putalpha(mask_pil)
            result = Image.alpha_composite(white_bg, img_pil)
            return result.convert('RGB')
        else:
            # Default: return RGBA
            img_pil = Image.fromarray(img_np).convert('RGBA')
            mask_pil = Image.fromarray(mask)
            img_pil.putalpha(mask_pil)
            return img_pil


def create_fast_remover(mode="base", jit=False, device=None, ckpt=None):
    """Factory function to create FastRemover instance."""
    return FastRemover(mode=mode, jit=jit, device=device, ckpt=ckpt)


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
        global _first_remover_init, _first_process_call

        total_start = time.time()
        debug_log("=" * 60)
        debug_log("InspyrenetRembg.remove_background() called")
        debug_log(f"  torchscript_jit: {torchscript_jit}")
        debug_log(f"  input images: {len(image)}")
        debug_log(f"  first_remover_init: {_first_remover_init}")
        debug_log(f"  first_process_call: {_first_process_call}")

        # Check for custom model path in ComfyUI models directory
        _t = time.time()
        custom_ckpt = get_model_path(mode="base")
        debug_log(f"get_model_path() returned: {custom_ckpt}", _t)

        # Use FAST REMOVER - bypasses the slow 68s initialization of standard Remover
        use_jit = torchscript_jit != 'default'

        _t = time.time()
        # FastRemover loads in ~1.5s vs 68s for standard Remover
        remover = create_fast_remover(mode="base", jit=use_jit, ckpt=custom_ckpt)
        debug_log("FastRemover initialized", _t)

        if _first_remover_init:
            debug_log("*** This was the FIRST Remover initialization ***")
            _first_remover_init = False

        # Log device info
        debug_log(f"  Remover device: {remover.device}")

        img_list = []
        for idx, img in enumerate(tqdm(image, "Inspyrenet Rembg")):
            _t = time.time()
            pil_img = tensor2pil(img)
            debug_log(f"  Image {idx}: tensor2pil conversion", _t) if idx == 0 else None

            _t = time.time()
            mid = remover.process(pil_img, type='rgba')
            if idx == 0:
                debug_log(f"  Image {idx}: remover.process() - FIRST IMAGE", _t)
                if _first_process_call:
                    debug_log("*** This was the FIRST process() call (may include CUDA kernel compilation) ***")
                    _first_process_call = False

            _t = time.time()
            out = pil2tensor(mid)
            debug_log(f"  Image {idx}: pil2tensor conversion", _t) if idx == 0 else None

            img_list.append(out)

        _t = time.time()
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        debug_log("Output tensor stacking completed", _t)

        debug_log(f"remove_background() TOTAL TIME", total_start)
        debug_log("=" * 60)

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
        global _first_remover_init, _first_process_call

        total_start = time.time()
        debug_log("=" * 60)
        debug_log("InspyrenetRembgAdvanced.remove_background() called")
        debug_log(f"  torchscript_jit: {torchscript_jit}")
        debug_log(f"  threshold: {threshold}")
        debug_log(f"  input images: {len(image)}")
        debug_log(f"  first_remover_init: {_first_remover_init}")
        debug_log(f"  first_process_call: {_first_process_call}")

        # Check for custom model path in ComfyUI models directory
        _t = time.time()
        custom_ckpt = get_model_path(mode="base")
        debug_log(f"get_model_path() returned: {custom_ckpt}", _t)

        # Use FAST REMOVER - bypasses the slow 68s initialization of standard Remover
        use_jit = torchscript_jit != 'default'

        _t = time.time()
        # FastRemover loads in ~1.5s vs 68s for standard Remover
        remover = create_fast_remover(mode="base", jit=use_jit, ckpt=custom_ckpt)
        debug_log("FastRemover initialized", _t)

        if _first_remover_init:
            debug_log("*** This was the FIRST Remover initialization ***")
            _first_remover_init = False

        # Log device info
        debug_log(f"  Remover device: {remover.device}")

        img_list = []
        for idx, img in enumerate(tqdm(image, "Inspyrenet Rembg")):
            _t = time.time()
            pil_img = tensor2pil(img)
            debug_log(f"  Image {idx}: tensor2pil conversion", _t) if idx == 0 else None

            _t = time.time()
            mid = remover.process(pil_img, type='rgba', threshold=threshold)
            if idx == 0:
                debug_log(f"  Image {idx}: remover.process() - FIRST IMAGE", _t)
                if _first_process_call:
                    debug_log("*** This was the FIRST process() call (may include CUDA kernel compilation) ***")
                    _first_process_call = False

            _t = time.time()
            out = pil2tensor(mid)
            debug_log(f"  Image {idx}: pil2tensor conversion", _t) if idx == 0 else None

            img_list.append(out)

        _t = time.time()
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        debug_log("Output tensor stacking completed", _t)

        debug_log(f"remove_background() TOTAL TIME", total_start)
        debug_log("=" * 60)

        return (img_stack, mask)