"""
real_movie_renderer.py
----------------------
Render the real Partaker tracking data as a GIF, frame by frame, so it can be
placed side-by-side with the simulator output.

    python real_movie_renderer.py path/to/enhanced_tracking_data_all_tracks.csv

Optional flags:
    --pixel_size      um/pixel (default 0.0645 for 100x Nikon confocal)
    --frame_interval  seconds between frames (default 300 = 5 min)
    --frame_max       last frame to render, inclusive (default 130)
    --iptg_frame      frame at which IPTG arrives, marked on output (default 28)
    --output          path to GIF (default out/real.gif)
    --fps             frames per second in the GIF (default 10)
"""

import argparse
import io
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from PIL import Image


def load_frames(csv_path, frame_max):
    df = pd.read_csv(csv_path)
    df = df[df.time_point <= frame_max]
    df = df[df.morphology_class != 'Artifact']
    df = df.dropna(subset=['x_um', 'y_um', 'major_axis_length',
                           'minor_axis_length', 'orientation_radians'])
    return df


def figure_extents(df, pad_um=2.0):
    x_min = float(df.x_um.min()) - pad_um
    x_max = float(df.x_um.max()) + pad_um
    y_min = float(df.y_um.min()) - pad_um
    y_max = float(df.y_um.max()) + pad_um
    return x_min, x_max, y_min, y_max


def render_frame(ax, df_t, pixel_size, extents, frame_idx, iptg_frame,
                 frame_interval):
    x_min, x_max, y_min, y_max = extents
    ax.clear()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # invert y so origin is top-left, matching image
    ax.set_aspect('equal')
    ax.set_facecolor('#0b0b0b')
    ax.set_xticks([])
    ax.set_yticks([])

    iptg_on = frame_idx >= iptg_frame
    edge = '#7df57d' if iptg_on else '#9ec5ff'
    face = '#244c24' if iptg_on else '#1f3a66'

    for _, row in df_t.iterrows():
        major_um = row.major_axis_length * pixel_size
        minor_um = row.minor_axis_length * pixel_size
        if not math.isfinite(major_um) or not math.isfinite(minor_um):
            continue
        if major_um <= 0 or minor_um <= 0:
            continue
        e = Ellipse(
            xy=(row.x_um, row.y_um),
            width=major_um,
            height=minor_um,
            angle=math.degrees(row.orientation_radians),
            facecolor=face,
            edgecolor=edge,
            linewidth=0.6,
        )
        ax.add_patch(e)

    t_seconds = frame_idx * frame_interval
    label = f"frame {frame_idx:>3} | t = {t_seconds/60:5.1f} min | n = {len(df_t)}"
    ax.text(0.02, 0.98, label,
            transform=ax.transAxes, ha='left', va='top',
            color='white', fontsize=9, family='monospace')

    if iptg_on:
        ax.text(0.98, 0.98, '+IPTG',
                transform=ax.transAxes, ha='right', va='top',
                color='#7df57d', fontsize=10, family='monospace',
                weight='bold')


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).convert('P', palette=Image.ADAPTIVE)


def main():
    parser = argparse.ArgumentParser(
        description='Render Partaker tracking CSV as a GIF')
    parser.add_argument('csv', help='Path to enhanced_tracking_data_all_tracks.csv')
    parser.add_argument('--pixel_size', type=float, default=0.0645)
    parser.add_argument('--frame_interval', type=float, default=300)
    parser.add_argument('--frame_max', type=int, default=130)
    parser.add_argument('--iptg_frame', type=int, default=28)
    parser.add_argument('--output', default='out/real.gif')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = load_frames(args.csv, args.frame_max)
    print(f"Kept {len(df)} cell-frames across {df.time_point.nunique()} frames")

    extents = figure_extents(df)
    print(f"Extents (um): x [{extents[0]:.1f}, {extents[1]:.1f}] "
          f"y [{extents[2]:.1f}, {extents[3]:.1f}]")

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#0b0b0b')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    frames = []
    for t in range(args.frame_max + 1):
        df_t = df[df.time_point == t]
        render_frame(ax, df_t, args.pixel_size, extents,
                     t, args.iptg_frame, args.frame_interval)
        frames.append(fig_to_pil(fig))
        if t % 10 == 0:
            print(f"  rendered frame {t}/{args.frame_max}")

    plt.close(fig)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    duration_ms = int(1000 / args.fps)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"\nDone! GIF: {args.output}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == '__main__':
    main()
