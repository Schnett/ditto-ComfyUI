import numpy as np
import cv2

from .loader import load_source_frames, check_resize
from .source2info import Source2Info


def _mean_filter(arr, k):
    n = arr.shape[0]
    half_k = k // 2
    res = []
    for i in range(n):
        s = max(0, i - half_k)
        e = min(n, i + half_k + 1)
        res.append(arr[s:e].mean(0))
    res = np.stack(res, 0)
    return res


def smooth_x_s_info_lst(x_s_info_list, ignore_keys=(), smo_k=13):
    keys = x_s_info_list[0].keys()
    N = len(x_s_info_list)
    smo_dict = {}
    for k in keys:
        _lst = [x_s_info_list[i][k] for i in range(N)]
        if k not in ignore_keys:
            _lst = np.stack(_lst, 0)
            _smo_lst = _mean_filter(_lst, smo_k)
        else:
            _smo_lst = _lst
        smo_dict[k] = _smo_lst

    smo_res = []
    for i in range(N):
        x_s_info = {k: smo_dict[k][i] for k in keys}
        smo_res.append(x_s_info)
    return smo_res


class AvatarRegistrar:
    """
    source image|video -> rgb_list -> source_info
    """
    def __init__(
        self,
        insightface_det_cfg,
        landmark106_cfg,
        landmark203_cfg,
        landmark478_cfg,
        appearance_extractor_cfg,
        motion_extractor_cfg,
    ):
        self.source2info = Source2Info(
            insightface_det_cfg,
            landmark106_cfg,
            landmark203_cfg,
            landmark478_cfg,
            appearance_extractor_cfg,
            motion_extractor_cfg,
        )

    def register(
        self,
        source,  # path | list[np.ndarray(H,W,3)] | np.ndarray[T,H,W,3] | np.ndarray[H,W,3]
        max_dim=1920,
        n_frames=-1,
        **kwargs,
    ):
        """
        kwargs:
            crop_scale: 2.3
            crop_vx_ratio: 0
            crop_vy_ratio: -0.125
            crop_flag_do_rot: True
        """
        # Accept either a file path or preloaded frames
        if isinstance(source, str):
            rgb_list, is_image_flag = load_source_frames(source, max_dim=max_dim, n_frames=n_frames)
        else:
            # Normalize frames to list of uint8 RGB arrays
            if isinstance(source, np.ndarray):
                if source.ndim == 3:
                    source = [source]
                elif source.ndim == 4:
                    source = [source[i] for i in range(source.shape[0])]
                else:
                    raise ValueError("Unsupported numpy array shape for frames")
            if not isinstance(source, (list, tuple)) or len(source) == 0:
                raise ValueError("source frames must be a non-empty list/array of images")

            frames = []
            for f in source:
                a = np.asarray(f)
                if a.ndim != 3 or a.shape[2] != 3:
                    raise ValueError("Each frame must be HxWx3 RGB")
                if a.dtype != np.uint8:
                    # Assume 0..1 floats and convert
                    a = (a.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
                frames.append(a)

            # Resize once consistently based on first frame
            h, w = frames[0].shape[:2]
            new_h, new_w, rsz_flag = check_resize(h, w, max_dim)
            if rsz_flag:
                frames = [cv2.resize(f, (new_w, new_h)) for f in frames]

            if n_frames > 0:
                frames = frames[:n_frames]
            rgb_list = frames
            is_image_flag = len(rgb_list) == 1
        source_info = {
            "x_s_info_lst": [],
            "f_s_lst": [],
            "M_c2o_lst": [],
            "eye_open_lst": [],
            "eye_ball_lst": [],
        }
        keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
        last_lmk = None
        for rgb in rgb_list:
            info = self.source2info(rgb, last_lmk, **kwargs)
            for k in keys:
                source_info[f"{k}_lst"].append(info[k])

            last_lmk = info["lmk203"]

        sc_f0 = source_info['x_s_info_lst'][0]['kp'].flatten()

        source_info["sc"] = sc_f0
        source_info["is_image_flag"] = is_image_flag
        source_info["img_rgb_lst"] = rgb_list

        return source_info
    
    def __call__(self, *args, **kwargs):
        return self.register(*args, **kwargs)
    