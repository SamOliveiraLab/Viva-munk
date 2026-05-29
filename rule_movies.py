"""
rule_movies.py
--------------
Render one standalone colored GIF per rule set, from the pickles the montage
scripts already saved. For talks: click through them one at a time and narrate
the story rule by rule (flow washes cells away -> attachment holds them ->
pressure slows growth to match reality).

No new simulation. Reads out/montage_pickles/montage_{default,hydro,attach,
pressure}.pkl and writes one GIF each.

Usage:
    python rule_movies.py \
        --pickle_dir out/runs/<stamp>_rule_ablation/pickles \
        --out_dir    out/runs/<stamp>_rule_ablation
"""

import argparse
import os
import pickle

from PIL import Image

from sim_mask_renderer import pick_frames_by_time, render_frame
from rule_montage import RULE_SETS


def render_movie(results, env_size, args, out_path):
    picks = pick_frames_by_time(results, args.frame_count, args.frame_interval)
    frames = []
    for i, state in enumerate(picks):
        img = render_frame(state, env_size, args.pixel_size_render,
                           color=True, label_ids=not args.no_ids,
                           font_size=args.font_size,
                           env_height_um=args.chamber_width)
        frames.append(img.convert('P', palette=Image.ADAPTIVE, colors=256))
        if i % 10 == 0:
            print(f"    frame {i}/{len(picks)-1}")
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    duration_ms = int(1000 / args.fps)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    print(f"  wrote {out_path}  ({len(frames)} frames @ {args.fps} fps)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pickle_dir', default='out/montage_pickles')
    p.add_argument('--out_dir', default='out')
    p.add_argument('--frame_count', type=int, default=28)
    p.add_argument('--frame_interval', type=float, default=300)
    p.add_argument('--pixel_size_render', type=float, default=0.0645)
    p.add_argument('--font_size', type=int, default=9)
    p.add_argument('--no_ids', action='store_true')
    p.add_argument('--chamber_width', type=float, default=None,
                   help='Real chamber width in um (y) for rectangular canvas. '
                        '52.5 for this chip. Omit for square.')
    p.add_argument('--fps', type=int, default=6)
    args = p.parse_args()

    for label, _flags, suffix in RULE_SETS:
        pk = os.path.join(args.pickle_dir, f'montage_{suffix}.pkl')
        if not os.path.exists(pk):
            print(f"Skipping {label}: no pickle at {pk} "
                  f"(run rule_montage.py first)")
            continue
        print(f"Rule set: {label}")
        with open(pk, 'rb') as f:
            payload = pickle.load(f)
        out_path = os.path.join(args.out_dir, f'rule_{suffix}.gif')
        render_movie(payload['results'], payload['env_size'], args, out_path)

    print("\nDone. One GIF per rule in", args.out_dir)


if __name__ == '__main__':
    main()
