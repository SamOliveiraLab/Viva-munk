"""
real_movie_renderer.py
----------------------
Render the real Partaker movie as a binary segmentation-mask GIF (white cells
on black background), reading directly from the omnipose label cache that
Partaker writes during tracking.

    python real_movie_renderer.py path/to/segmentation_cache.h5

The default cache is at:
    /Volumes/SAM1/server_workspace_backup/Bukola/partaker_dataset_DT/
        Saved_experiment_w_tracking/segmentation_cache.h5

Inside the cache, the array `mmap_arrays/omnipose_bact_phase/array` has shape
(T, P, C, H, W) with dtype uint16. Channel 0 is the instance-label mask;
threshold > 0 to get a binary cell mask.

Optional flags:
    --position    position index in the multi-position acquisition (default 0)
    --channel     channel index of the label mask (default 0)
    --frame_max   last frame to include, inclusive (default 130)
    --output      path to GIF (default out/real.gif)
    --fps         frames per second in the GIF (default 10)
"""

import argparse
import os

import h5py
import numpy as np
from PIL import Image


DEFAULT_CACHE = (
    '/Volumes/SAM1/server_workspace_backup/Bukola/partaker_dataset_DT/'
    'Saved_experiment_w_tracking/segmentation_cache.h5'
)
DEFAULT_ROI = (
    '/Volumes/SAM1/server_workspace_backup/Bukola/partaker_dataset_DT/'
    'Saved_experiment_w_tracking/roi_mask.npy'
)
ARRAY_KEY = 'mmap_arrays/omnipose_bact_phase/array'


def load_mask_frames(h5_path, position, channel, frame_max, roi_mask=None):
    with h5py.File(h5_path, 'r') as f:
        arr = f[ARRAY_KEY]
        n_frames_avail = arr.shape[0]
        last = min(frame_max, n_frames_avail - 1)
        frames = []
        for t in range(last + 1):
            labels = arr[t, position, channel]
            mask = (labels > 0).astype(np.uint8) * 255
            if roi_mask is not None:
                mask = np.where(roi_mask > 0, mask, 0).astype(np.uint8)
            frames.append(mask)
            if t % 10 == 0:
                print(f"  loaded frame {t}/{last}")
        return frames


def crop_to_roi_bbox(frames, roi_mask, pad=4):
    ys, xs = np.where(roi_mask > 0)
    if len(ys) == 0:
        return frames
    y0 = max(0, ys.min() - pad)
    y1 = min(roi_mask.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(roi_mask.shape[1], xs.max() + pad + 1)
    return [f[y0:y1, x0:x1] for f in frames]


def main():
    parser = argparse.ArgumentParser(
        description='Render real-movie binary masks from a Partaker '
                    'segmentation cache as a GIF')
    parser.add_argument('h5', nargs='?', default=DEFAULT_CACHE,
                        help='Path to segmentation_cache.h5')
    parser.add_argument('--position', type=int, default=0)
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--frame_max', type=int, default=130)
    parser.add_argument('--output', default='out/real.gif')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--roi_mask', default=DEFAULT_ROI,
                        help='Path to roi_mask.npy; pass "" to skip ROI cropping')
    args = parser.parse_args()

    print(f"Reading masks from {args.h5}")
    print(f"  position={args.position}  channel={args.channel}  "
          f"frame_max={args.frame_max}")

    roi = None
    if args.roi_mask:
        roi = np.load(args.roi_mask)
        print(f"Applied ROI mask {args.roi_mask}  shape={roi.shape}  "
              f"keep_frac={float((roi>0).mean()):.3f}")

    masks = load_mask_frames(args.h5, args.position, args.channel,
                             args.frame_max, roi_mask=roi)
    if roi is not None:
        masks = crop_to_roi_bbox(masks, roi)
    print(f"Loaded {len(masks)} mask frames; size {masks[0].shape}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    pil_frames = [Image.fromarray(m, mode='L').convert('P') for m in masks]
    duration_ms = int(1000 / args.fps)
    pil_frames[0].save(
        args.output,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"\nDone! GIF: {args.output}  "
          f"({len(pil_frames)} frames @ {args.fps} fps)")


if __name__ == '__main__':
    main()
