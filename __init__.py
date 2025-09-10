import os
import sys
import time
import json
import math
import librosa
import numpy as np
import torch

BASE_DIR = os.path.dirname(__file__)

# Prefer package-relative import; fallback to absolute by amending sys.path
try:
    from .stream_pipeline_offline import StreamSDK
except Exception:
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    from stream_pipeline_offline import StreamSDK


DEFAULT_DATA_ROOT = os.path.join(BASE_DIR, "checkpoints", "ditto_pytorch")
DEFAULT_CFG_PKL = os.path.join(BASE_DIR, "checkpoints", "ditto_cfg", "v0.4_hubert_cfg_pytorch.pkl")


def _parse_emo(arg: str | None):
    if not arg:
        return None
    name_to_idx = {
        "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
        "neutral": 4, "sad": 5, "surprise": 6, "contempt": 7,
    }
    parts = [p.strip().lower() for p in arg.split(',') if p.strip()]
    idxs = []
    for p in parts:
        if p.isdigit():
            i = int(p)
            if 0 <= i <= 7:
                idxs.append(i)
        elif p in name_to_idx:
            idxs.append(name_to_idx[p])
    if not idxs:
        return None
    return idxs[0] if len(idxs) == 1 else idxs


class DittoTalkingHead:
    CATEGORY = "Ditto"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "", "multiline": False}),
                "source_images": ("IMAGE", {}),
                "data_root": ("STRING", {"default": DEFAULT_DATA_ROOT, "multiline": False}),
                "cfg_pkl": ("STRING", {"default": DEFAULT_CFG_PKL, "multiline": False}),
            },
            "optional": {
                "source_path": ("STRING", {"default": "", "multiline": False}),
                "personalized_model_path": ("STRING", {"default": "", "multiline": False}),
                "emo": ("STRING", {"default": "neutral", "multiline": False}),
                "mouth_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "head_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "smooth_motion_k": ("INT", {"default": 13, "min": 1, "max": 30, "step": 1}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def run(
        self,
        audio_path: str,
        source_images,  # ComfyUI IMAGE tensor: [T,H,W,C] in 0..1
        data_root: str,
        cfg_pkl: str,
        source_path: str = "",
        personalized_model_path: str = "",
        emo: str = "neutral",
        mouth_scale: float = 1.0,
        head_scale: float = 1.0,
        smooth_motion_k: int = 13,
        output_path: str = "",
    ):
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"audio_path not found: {audio_path}")

        # Prepare source frames: prefer provided IMAGE tensor; fallback to source_path
        source = None
        img_base_for_name = ""
        if source_images is not None:
            # Expect torch.Tensor [T,H,W,C] or [1,H,W,C], float32 0..1
            if isinstance(source_images, torch.Tensor):
                t = source_images
            else:
                # Some nodes might wrap; try to convert
                try:
                    t = torch.tensor(source_images)
                except Exception:
                    raise TypeError("source_images must be a torch.Tensor or tensor-like array")
            if t.dim() == 3:
                t = t.unsqueeze(0)
            np_imgs = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            # Convert to list of HxWx3 RGB frames
            source = [np_imgs[i] for i in range(np_imgs.shape[0])]
            img_base_for_name = f"frames{len(source)}"
        elif source_path:
            if not os.path.isfile(source_path):
                raise FileNotFoundError(f"source_path not found: {source_path}")
            source = source_path
            img_base_for_name = os.path.splitext(os.path.basename(source_path))[0]
        else:
            raise ValueError("Provide either source_images (IMAGE) or a valid source_path.")

        setup_kwargs = {}
        if personalized_model_path:
            setup_kwargs["personalized_model_path"] = personalized_model_path

        emo_val = _parse_emo(emo)
        if emo_val is not None:
            setup_kwargs["emo"] = emo_val

        if isinstance(smooth_motion_k, int) and smooth_motion_k > 0:
            setup_kwargs["smo_k_d"] = int(smooth_motion_k)

        use_d_keys = {}
        if mouth_scale != 1.0:
            use_d_keys["exp"] = float(mouth_scale)
        if head_scale != 1.0:
            for k in ("pitch", "yaw", "roll"):
                use_d_keys[k] = float(head_scale)
        if use_d_keys:
            setup_kwargs["use_d_keys"] = use_d_keys

        if not output_path:
            base_out = os.path.join(os.getcwd(), "output")
            os.makedirs(base_out, exist_ok=True)
            # Extract base names without extension
            audio_base = os.path.splitext(os.path.basename(audio_path))[0]
            img_base = img_base_for_name or "image"
            emo_str = str(emo) if emo else "neutral"
            m_str = f"m{mouth_scale}" if mouth_scale is not None else "m1.0"
            h_str = f"h{head_scale}" if head_scale is not None else "h1.0"
            model_base = os.path.splitext(os.path.basename(personalized_model_path))[0] if personalized_model_path else "none"
            # Sanitize for filename (remove problematic chars)
            def safe(s):
                return str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")
            fname = f"{safe(audio_base)}_{safe(img_base)}_{safe(emo_str)}_{m_str}_{h_str}_{safe(model_base)}.mp4"
            output_path = os.path.join(base_out, fname)

        SDK = StreamSDK(cfg_pkl, data_root)
        SDK.setup(source, output_path, **setup_kwargs)

        audio, sr = librosa.core.load(audio_path, sr=16000)
        num_f = math.ceil(len(audio) / 16000 * 25)
        SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})

        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
        SDK.close()

        cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
        os.system(cmd)

        try:
            if os.path.exists(SDK.tmp_output_path):
                os.remove(SDK.tmp_output_path)
        except Exception:
            pass

        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "DittoTalkingHead": DittoTalkingHead,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DittoTalkingHead": "Ditto Talking Head",
}
