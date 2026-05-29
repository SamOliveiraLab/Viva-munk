"""
rule_montage.py
---------------
Supplementary small-multiples figure: one row of panels, each the SAME late
frame under a different rule set, with the real frame as the anchor at the end.
Shows at a glance how each rule bends the colony toward reality. The visual
companion to the divergence plot.

Panels (left to right):
    defaults | + hydro | + hydro + attach | + hydro + attach + pressure | REAL

All sim panels are seeded with the same full frame-0 population (--max_cells),
so the comparison is fair. Each sim is pickled so re-renders skip the sim.

Usage:
    python rule_montage.py \
        --sim_csv /Volumes/SAM1/server_workspace_backup/cell_history_amby.csv \
        --h5 /Volumes/SAM1/.../segmentation_cache.h5 \
        --roi_mask /Volumes/SAM1/.../roi_mask.npy \
        --frame 27 --max_cells 250 \
        --out out/rule_montage.png
"""

import argparse
import os
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_mask_renderer import run_sim, pick_frames_by_time, render_frame
from real_movie_renderer import load_mask_frames, crop_to_roi_bbox


# Rule sets in order. Each is (label, dict-of-flags, pickle-suffix).
RULE_SETS = [
    ('defaults',                 dict(hydro=False, attach=False, pressure=False), 'default'),
    ('+ hydro',                  dict(hydro=True,  attach=False, pressure=False), 'hydro'),
    ('+ hydro + attach',         dict(hydro=True,  attach=True,  pressure=False), 'attach'),
    ('+ hydro + attach + press', dict(hydro=True,  attach=True,  pressure=True),  'pressure'),
]


def _load_font(size):
    for path in (
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/Library/Fonts/Arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_sim_panel(args, flags, pickle_path, font_size):
    """Run (or reuse) one sim and return the chosen frame as an RGB PIL image."""
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

    # pick the frame at args.frame (frame_interval spacing matches real cadence)
    picks = pick_frames_by_time(results, args.frame + 1, args.frame_interval)
    state = picks[min(args.frame, len(picks) - 1)]
    img = render_frame(state, env_size, args.pixel_size_render,
                       color=True, label_ids=not args.no_ids,
                       font_size=font_size,
                       env_height_um=args.chamber_width)
    return img.convert('RGB')


def get_real_panel(args, font_size):
    """Load the real frame at args.frame as an RGB PIL image, ROI-cropped."""
    roi = None
    if args.roi_mask:
        roi = np.load(args.roi_mask)
    frames = load_mask_frames(
        args.h5, args.position, args.channel, args.frame,
        roi_mask=roi, color=True, label_ids=not args.no_ids,
        font_size=font_size,
    )
    if roi is not None:
        frames = crop_to_roi_bbox(frames, roi)
    real = frames[min(args.frame, len(frames) - 1)]
    return Image.fromarray(real, mode='RGB')


def resize_to_height(img, target_h):
    w, h = img.size
    new_w = max(1, int(round(w * target_h / h)))
    return img.resize((new_w, target_h), Image.LANCZOS)


def tile_with_labels(panels, labels, label_h=34, gap=8, bg=(0, 0, 0)):
    """Tile RGB panels horizontally, each with a label strip underneath."""
    target_h = max(p.size[1] for p in panels)
    panels = [resize_to_height(p, target_h) for p in panels]
    font = _load_font(20)

    total_w = sum(p.size[0] for p in panels) + gap * (len(panels) - 1)
    total_h = target_h + label_h
    canvas = Image.new('RGB', (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    x = 0
    for panel, label in zip(panels, labels):
        canvas.paste(panel, (x, 0))
        # center the label text under the panel
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(label) * 10
        tx = x + (panel.size[0] - tw) // 2
        ty = target_h + (label_h - 20) // 2
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
        x += panel.size[0] + gap
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sim_csv', required=True)
    p.add_argument('--h5', required=True, help='segmentation_cache.h5 for the real frame')
    p.add_argument('--roi_mask', default='', help='roi_mask.npy; "" to skip cropping')
    p.add_argument('--position', type=int, default=0)
    p.add_argument('--channel', type=int, default=0)
    p.add_argument('--frame', type=int, default=27,
                   help='Frame index to show in every panel (default 27, last pre-IPTG)')
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
    p.add_argument('--pickle_dir', default='out/montage_pickles')
    p.add_argument('--reuse_pickles', action='store_true',
                   help='Skip sims that already have a pickle in --pickle_dir')
    p.add_argument('--out', default='out/rule_montage.png')
    args = p.parse_args()

    os.makedirs(args.pickle_dir, exist_ok=True)
    panels = []
    labels = []
    for label, flags, suffix in RULE_SETS:
        print(f"Rule set: {label}")
        pk = os.path.join(args.pickle_dir, f'montage_{suffix}.pkl')
        panels.append(render_sim_panel(args, flags, pk, args.font_size))
        labels.append(label)

    print("Real frame:")
    panels.append(get_real_panel(args, args.font_size))
    labels.append(f'REAL (frame {args.frame})')

    print("Tiling ...")
    montage = tile_with_labels(panels, labels)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    montage.save(args.out)
    print(f"\nDone! Montage: {args.out}  ({montage.size[0]}x{montage.size[1]})")


if __name__ == '__main__':
    main()
