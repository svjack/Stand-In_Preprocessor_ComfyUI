import os
import cv2
import requests
import torch
import numpy as np
import PIL.Image
import PIL.ImageOps
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from torchvision.transforms.functional import normalize
from typing import Union, Optional
import folder_paths
from huggingface_hub import snapshot_download

LOADED_PROCESSORS = {}

def set_extra_config_model_path(extra_config_models_dir_key, models_dir_name: str):
    models_dir_default = os.path.join(folder_paths.models_dir, models_dir_name)
    if extra_config_models_dir_key not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[extra_config_models_dir_key] = (
            [os.path.join(folder_paths.models_dir, models_dir_name)],
            folder_paths.supported_pt_extensions,
        )
    else:
        if not os.path.exists(models_dir_default):
            os.makedirs(models_dir_default, exist_ok=True)
        folder_paths.add_model_folder_path(extra_config_models_dir_key, models_dir_default, is_default=True)

set_extra_config_model_path("insightface", "insightface")

def _img2tensor(img: np.ndarray, bgr2rgb: bool = True) -> torch.Tensor:
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)

def _pad_to_square(img: np.ndarray, pad_color: int = 255) -> np.ndarray:
    h, w, _ = img.shape
    if h == w:
        return img

    if h > w:
        pad_size = (h - w) // 2
        padded_img = cv2.copyMakeBorder(
            img, 0, 0, pad_size, h - w - pad_size, cv2.BORDER_CONSTANT, value=[pad_color] * 3,
        )
    else:
        pad_size = (w - h) // 2
        padded_img = cv2.copyMakeBorder(
            img, pad_size, w - h - pad_size, 0, 0, cv2.BORDER_CONSTANT, value=[pad_color] * 3,
        )
    return padded_img

def tensor_to_cv2_img(tensor: torch.Tensor) -> np.ndarray:
    img_np = tensor.squeeze(0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr

def cv2_img_to_tensor(img: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).unsqueeze(0)


class FaceProcessorLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "antelopev2_path": ("STRING", {"default": "models/insightface"})
            }
        }

    RETURN_TYPES = ("FACE_PROCESSOR",)
    RETURN_NAMES = ("face_processor",)
    FUNCTION = "load_processor"
    CATEGORY = "Stand-In"

    def load_processor(self, antelopev2_path="models/insightface"):
        if "face_processor" in LOADED_PROCESSORS:
            print("Reusing cached FaceProcessor models.")
            return (LOADED_PROCESSORS["face_processor"],)
        antelopev2_path = folder_paths.get_folder_paths("insightface")[0]
        print("Initializing FaceProcessor models...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        
        if not os.path.exists(antelopev2_path):
            os.makedirs(antelopev2_path)
            print(f"Created insightface model directory at: {antelopev2_path}")
        try:
            print("Attempting to load FaceAnalysis model from local files...")
            app = FaceAnalysis(name="antelopev2", root=antelopev2_path, providers=providers)
        except Exception as e:
            print(f"Failed to load local FaceAnalysis model: {e}")
            if snapshot_download is None:
                raise ImportError("huggingface_hub is not installed. Please install it with 'pip install huggingface_hub' to enable automatic model downloading.") from e

            print("Attempting to download model from Hugging Face...")
            download_target_path = os.path.join(antelopev2_path, "models", "antelopev2")
            snapshot_download(
                repo_id="DIAMONIK7777/antelopev2",
                local_dir=download_target_path,
            )
            print("Model downloaded. Retrying to load FaceAnalysis model...")
            app = FaceAnalysis(name="antelopev2", root=antelopev2_path, providers=providers)

        app.prepare(ctx_id=0, det_size=(640, 640))

        parsing_model = init_parsing_model(model_name="bisenet", device=device)
        parsing_model.eval()

        processor_tuple = (app, parsing_model, device)
        LOADED_PROCESSORS["face_processor"] = processor_tuple
        
        print("FaceProcessor models initialized and cached successfully.")
        
        return (processor_tuple,)



class ApplyFaceProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_processor": ("FACE_PROCESSOR",),
                "image": ("IMAGE",),
                "resize_to": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "border_thresh": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "face_crop_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_processing"
    CATEGORY = "FaceUtils"

    def apply_processing(self, face_processor, image, resize_to, border_thresh, face_crop_scale):
        app, parsing_model, device = face_processor
        
        frame = tensor_to_cv2_img(image)

        faces = app.get(frame)
        h, w, _ = frame.shape
        image_to_process = None

        if not faces:
            print("[Warning] No face detected. Using the whole image, padded to square.")
            image_to_process = _pad_to_square(frame, pad_color=255)
        else:
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = map(int, largest_face.bbox)

            is_close_to_border = (x1 <= border_thresh and y1 <= border_thresh and 
                                  x2 >= w - border_thresh and y2 >= h - border_thresh)

            if is_close_to_border:
                print("[Info] Face is close to border, padding original image to square.")
                image_to_process = _pad_to_square(frame, pad_color=255)
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                side = int(max(x2 - x1, y2 - y1) * face_crop_scale)
                half = side // 2

                left, top = max(cx - half, 0), max(cy - half, 0)
                right, bottom = min(cx + half, w), min(cy + half, h)

                cropped_face = frame[top:bottom, left:right]
                image_to_process = _pad_to_square(cropped_face, pad_color=255)

        image_resized = cv2.resize(image_to_process, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

        face_tensor = _img2tensor(image_resized, bgr2rgb=True).unsqueeze(0).to(device)
        with torch.no_grad():
            normalized_face = normalize(face_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            parsing_out = parsing_model(normalized_face)[0]
            parsing_mask = parsing_out.argmax(dim=1, keepdim=True)

        mask = (parsing_mask.squeeze().cpu().numpy() == 0).astype(np.uint8)
        white_background = np.ones_like(image_resized, dtype=np.uint8) * 255
        mask_3channel = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
        result_img_bgr = np.where(mask_3channel == 255, white_background, image_resized)

        result_tensor = cv2_img_to_tensor(result_img_bgr)

        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "FaceProcessorLoader": FaceProcessorLoader,
    "ApplyFaceProcessor": ApplyFaceProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceProcessorLoader": "Stand-In Processor Loader",
    "ApplyFaceProcessor": "Apply Stand-In Processor"
}