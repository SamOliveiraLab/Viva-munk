"""
tracking_movie_renderer.py
--------------------------
Render the Partaker tracking CSV as a binary mask GIF — each frame shows
every tracked cell as a capsule drawn at its tracked position/size/angle.

Use this to sanity-check tracking against the raw segmentation masks
(real.gif): if tracking is faithful, the two GIFs should look the same.

    python tracking_movie_renderer.py path/to/cell_history.csv

Optional flags:
    --roi_mask    path to roi_mask.npy ('' to skip ROI crop)
    --frame_max   last frame inclusive (default 130)
    --output      path to GIF (default out/tracking.gif)
    --fps         frames per second (default 10)
"""

import argparse
import csv
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from partaker_to_vivamunk import parse_array
from real_movie_renderer import crop_to_roi_bbox
from sim_mask_renderer import draw_capsule


DEFAULT_CSV = '/Volumes/SAM1/server_workspace_backup/cell_history_amby.csv'
DEFAULT_ROI = (
    '/Volumes/SAM1/server_workspace_backup/Bukola/partaker_dataset_DT/'
    'Saved_experiment_w_tracking/roi_mask.npy'
)


def load_cell_biographies(csv_path):
    cells = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start_time = int(row['start_time'])
            except (KeyError, ValueError):
                continue
            xs = parse_array(row['x_positions'])
            ys = parse_array(row['y_positions'])
            ls = parse_array(row['lengths'])
            ws = parse_array(row['widths'])
            ang = parse_array(row['orientations'])
            morphs = (row.get('morphology_classes', '') or '').split(',')
            n = min(len(xs), len(ys), len(ls), len(ws), len(ang))
            if n == 0:
                continue
            cells.append({
                'start_time': start_time,
                'x': xs[:n], 'y': ys[:n],
                'length': ls[:n], 'width': ws[:n],
                'angle': ang[:n], 'morphs': morphs,
            })
    return cells


def render_tracking_frame(cells, frame_idx, canvas_shape):
    H, W = canvas_shape
    img = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(img)
    drawn = 0
    for c in cells:
        k = frame_idx - c['start_time']
        if k < 0 or k >= len(c['x']):
            continue
        if k < len(c['morphs']) and c['morphs'][k].strip() == 'Artifact':
            continue
        cx, cy = c['x'][k], c['y'][k]
        length, width, angle = c['length'][k], c['width'][k], c['angle'][k]
        if not all(map(math.isfinite, [cx, cy, length, width, angle])):
            continue
        radius = max(width / 2.0, 0.5)
        length = max(length, radius * 2.0)
        draw_capsule(draw, cx, cy, length, radius, angle, fill=255)
        drawn += 1
    return np.array(img, dtype=np.uint8), drawn


def main():
    parser = argparse.ArgumentParser(
        description='Render Partaker tracking CSV as a binary mask GIF')
    parser.add_argument('csv', nargs='?', default=DEFAULT_CSV)
    parser.add_argument('--roi_mask', default=DEFAULT_ROI,
                        help='Path to roi_mask.npy; pass "" to skip ROI crop')
    parser.add_argument('--frame_max', type=int, default=130)
    parser.add_argument('--output', default='out/tracking.gif')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    print(f"Loading tracking biographies from {args.csv}")
    cells = load_cell_biographies(args.csv)
    print(f"  {len(cells)} cells loaded")

    roi = None
    if args.roi_mask:
        roi = np.load(args.roi_mask)
        canvas_shape = roi.shape
        print(f"ROI mask {args.roi_mask}  shape={roi.shape}")
    else:
        all_x = [x for c in cells for x in c['x'] if math.isfinite(x)]
        all_y = [y for c in cells for y in c['y'] if math.isfinite(y)]
        canvas_shape = (int(max(all_y)) + 32, int(max(all_x)) + 32)
        print(f"No ROI; canvas inferred {canvas_shape}")

    frames = []
    for t in range(args.frame_max + 1):
        arr, n_drawn = render_tracking_frame(cells, t, canvas_shape)
        if roi is not None:
            arr = np.where(roi > 0, arr, 0).astype(np.uint8)
        frames.append(arr)
        if t % 10 == 0:
            print(f"  rendered frame {t}/{args.frame_max}  ({n_drawn} cells)")

    if roi is not None:
        frames = crop_to_roi_bbox(frames, roi)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    pil_frames = [Image.fromarray(f, mode='L').convert('P') for f in frames]
    duration_ms = int(1000 / args.fps)
    pil_frames[0].save(
        args.output, save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    print(f"\nDone! GIF: {args.output}  "
          f"({len(pil_frames)} frames @ {args.fps} fps)")


if __name__ == '__main__':
    main()
