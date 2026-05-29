"""
rule_montage_movie.py
---------------------
Animated companion to rule_montage.py. Same five panels
(defaults | +hydro | +hydro+attach | +hydro+attach+pressure | REAL) but as a
single GIF where every panel plays in sync through the pre-IPTG window. The
eye watches each rule diverge from (or track) reality frame by frame.

All sim panels seeded with the same full frame-0 population (--max_cells).
Each sim is pickled so re-renders skip the sim (--reuse_pickles).

Usage:
    python rule_montage_movie.py \
        --sim_csv /Volumes/SAM1/.../cell_history_amby.csv \
        --h5 /Volumes/SAM1/.../segmentation_cache.h5 \
        --roi_mask /Volumes/SAM1/.../roi_mask.npy \
        --frame_count 28 --max_cells 250 \
        --out out/rule_montage_movie.gif
"""

import argparse
import os
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_mask_renderer import run_sim, pick_frames_by_time, render_frame
from real_movie_renderer import load_mask_frames, crop_to_roi_bbox
from rule_montage import RULE_SETS, _load_font, resize_to_height


def sim_frames(args, flags, pickle_path):
    """Run (or reuse) one sim, return list of RGB PIL frames over the window."""
    if args.reuse_pickles and os.path.exists(pickle_path):
        print(f"  reuse {pickle_path}")
        with open(pickle_path, 'rb') as f:
            payload = pickle.load(f)
        results, env_size = payload['results'], payload['env_size']
    else:
        results, env_size = run_sim(
            args.sim_csv, args.pixel_size, args.frame_interval,
            args.max_cells, args.sim_time,
            chamber_length=args.chamber_length,
            chamber_width=args.chamber_width,
            **flags,
        )
        os.makedirs(os.path.dirname(pickle_path) or '.', exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump({'results': results, 'env_size': env_size}, f)
        print(f"  pickled {pickle_path}")

    picks = pick_frames_by_time(results, args.frame_count, args.frame_interval)
    frames = []
    for state in picks:
        img = render_frame(state, env_size, args.pixel_size_render,
                           color=True, label_ids=not args.no_ids,
                           font_size=args.font_size,
                           env_height_um=args.chamber_width)
        frames.append(img.convert('RGB'))
    return frames


def real_frames(args):
    roi = None
    if args.roi_mask:
        roi = np.load(args.roi_mask)
    frames = load_mask_frames(
        args.h5, args.position, args.channel, args.frame_count - 1,
        roi_mask=roi, color=True, label_ids=not args.no_ids,
        font_size=args.font_size,
    )
    if roi is not None:
        frames = crop_to_roi_bbox(frames, roi)
    return [Image.fromarray(f, mode='RGB') for f in frames]


def compose_row(panels_at_t, labels, target_h, label_h=34, gap=8, bg=(0, 0, 0)):
    panels = [resize_to_height(p, target_h) for p in panels_at_t]
    font = _load_font(20)
    total_w = sum(p.size[0] for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new('RGB', (total_w, target_h + label_h), bg)
    draw = ImageDraw.Draw(canvas)
    x = 0
    for panel, label in zip(panels, labels):
        canvas.paste(panel, (x, 0))
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(label) * 10
        draw.text((x + (panel.size[0] - tw) // 2, target_h + (label_h - 20) // 2),
                  label, fill=(255, 255, 255), font=font)
        x += panel.size[0] + gap
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sim_csv', required=True)
    p.add_argument('--h5', required=True)
    p.add_argument('--roi_mask', default='')
    p.add_argument('--position', type=int, default=0)
    p.add_argument('--channel', type=int, default=0)
    p.add_argument('--frame_count', type=int, default=28,
                   help='Frames to animate (default 28 = pre-IPTG window)')
    p.add_argument('--pixel_size', type=float, default=0.0645)
    p.add_argument('--pixel_size_render', type=float, default=0.0645)
    p.add_argument('--frame_interval', type=float, default=300)
    p.add_argument('--sim_time', type=float, default=8100)
    p.add_argument('--max_cells', type=int, default=250)
    p.add_argument('--chamber_length', type=float, default=None,
                   help='Real chamber length in um (x). 70 for this chip.')
    p.add_argument('--chamber_width', type=float, default=None,
                   help='Real chamber width in um (y). 52.5 for this chip.')
    p.add_argument('--font_size', type=int, default=9)
    p.add_argument('--no_ids', action='store_true')
    p.add_argument('--fps', type=int, default=6)
    p.add_argument('--pickle_dir', default='out/montage_pickles')
    p.add_argument('--reuse_pickles', action='store_true')
    p.add_argument('--out', default='out/rule_montage_movie.gif')
    args = p.parse_args()

    os.makedirs(args.pickle_dir, exist_ok=True)

    # Collect per-panel frame lists
    column_frames = []
    labels = []
    for label, flags, suffix in RULE_SETS:
        print(f"Rule set: {label}")
        pk = os.path.join(args.pickle_dir, f'montage_{suffix}.pkl')
        column_frames.append(sim_frames(args, flags, pk))
        labels.append(label)

    print("Real frames:")
    column_frames.append(real_frames(args))
    labels.append('REAL')

    # Align frame counts (use the shortest so every panel has a frame at each t)
    n = min(len(c) for c in column_frames)
    print(f"Animating {n} synced frames across {len(column_frames)} panels")

    target_h = max(c[0].size[1] for c in column_frames)
    row_frames = []
    for t in range(n):
        panels_at_t = [c[t] for c in column_frames]
        row = compose_row(panels_at_t, labels, target_h)
        row_frames.append(row.convert('P', palette=Image.ADAPTIVE, colors=256))
        if t % 5 == 0:
            print(f"  composed frame {t}/{n-1}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    duration_ms = int(1000 / args.fps)
    row_frames[0].save(
        args.out, save_all=True, append_images=row_frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    print(f"\nDone! GIF: {args.out}  ({n} frames @ {args.fps} fps, "
          f"{row_frames[0].size[0]}x{row_frames[0].size[1]})")


if __name__ == '__main__':
    main()
