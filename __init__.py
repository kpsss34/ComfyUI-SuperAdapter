import os
import torch
import folder_paths
import comfy.utils
from safetensors.torch import load_file

adapter_dir = os.path.join(folder_paths.models_dir, "super_adapter")
if not os.path.exists(adapter_dir):
    os.makedirs(adapter_dir)

folder_paths.folder_names_and_paths["super_adapter"] = (
    [adapter_dir],
    {".safetensors", ".pt", ".bin"}
)

class ApplySuperAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "adapter_name": (folder_paths.get_filename_list("super_adapter"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply_super_adapter"
    CATEGORY = "Flux/Super Adapter"

    def apply_super_adapter(self, model, adapter_name, strength):
        if strength == 0.0:
            return (model,)

        adapter_path = folder_paths.get_full_path("super_adapter", adapter_name)
        print(f"[Super Adapter] Loading Power from: {adapter_path}")

        if adapter_path.endswith(".safetensors"):
            sd = load_file(adapter_path)
        else:
            sd = torch.load(adapter_path, map_location="cpu")

        patches = {}
        for key, delta in sd.items():
            comfy_key = f"diffusion_model.{key}" if not key.startswith("diffusion_model.") else key
            patches[comfy_key] = (delta * strength,)

        m = model.clone()
        m.add_patches(patches, strength_patch=1.0, strength_model=1.0)

        print(f"[Super Adapter] Successfully injected {len(patches)} layer modifications!")
        return (m,)

NODE_CLASS_MAPPINGS = {
    "ApplySuperAdapter": ApplySuperAdapter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplySuperAdapter": "Apply Super Adapter (Flux/SDXL)"
}
