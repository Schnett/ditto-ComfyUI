import librosa
import math
import os
import numpy as np
import random
import torch
import pickle

from stream_pipeline_offline import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source_path, output_path, **setup_kwargs)

    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)

    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_pytorch/", help="path to trt data_root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", help="path to cfg_pkl")

    parser.add_argument("--audio_path", type=str, help="path to input wav")
    parser.add_argument("--source_path", type=str, help="path to input image")
    parser.add_argument("--output_path", type=str, help="path to output mp4")
    parser.add_argument("--personalized_model_path", type=str, default=None, help="Path to custom fine-tuned model (optional)")
    # Keep only the essential expressiveness controls
    parser.add_argument("--emo", type=str, default=None, help="Emotion(s): index 0-7 or names: angry,disgust,fear,happy,neutral,sad,surprise,contempt. neutral default")
    parser.add_argument("--mouth_scale", type=float, default=None, help="Scale mouth/expression (exp). 1.0 default")
    parser.add_argument("--head_scale", type=float, default=None, help="Uniform scale for head pose (pitch,yaw,roll). 1.0 default")
    parser.add_argument("--smooth_motion_k", type=int, default=None, help="Temporal smoothing kernel for dynamics (smo_k_d). Lower values = more movement. Higher values = less movement. 13.0 default")
    args = parser.parse_args()

    # init sdk
    data_root = args.data_root   # model dir
    cfg_pkl = args.cfg_pkl     # cfg pkl
    SDK = StreamSDK(cfg_pkl, data_root)

    # input args
    audio_path = args.audio_path    # .wav
    source_path = args.source_path   # video|image
    output_path = args.output_path   # .mp4

    # Build setup/run kwargs from CLI
    setup_kwargs = {}
    if args.personalized_model_path:
        setup_kwargs["personalized_model_path"] = args.personalized_model_path

    # Emotion parsing
    def parse_emo(arg: str | None):
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

    emo_val = parse_emo(args.emo)
    if emo_val is not None:
        setup_kwargs["emo"] = emo_val

    # Motion smoothing control
    if args.smooth_motion_k is not None:
        setup_kwargs["smo_k_d"] = int(args.smooth_motion_k)

    # Movement scales via use_d_keys
    use_d_keys = {}
    if args.mouth_scale is not None:
        use_d_keys["exp"] = float(args.mouth_scale)
    if args.head_scale is not None:
        for k in ("pitch", "yaw", "roll"):
            use_d_keys[k] = float(args.head_scale)
    if use_d_keys:
        setup_kwargs["use_d_keys"] = use_d_keys

    # run
    # seed_everything(1024)
    # Pass setup_kwargs via more_kwargs
    run(SDK, audio_path, source_path, output_path, more_kwargs={"setup_kwargs": setup_kwargs})
