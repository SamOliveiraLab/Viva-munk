"""
sim_mask_renderer.py
--------------------
Run the Partaker-seeded smoke sim and render a binary segmentation-mask GIF
(white capsules on black) that matches the look of `real.gif`. Also pickles
the raw timeseries to `<output_pickle>` so future re-renders skip the sim.

Usage:
    python sim_mask_renderer.py path/to/cell_history.csv

Optional flags mirror partaker_to_vivamunk.py plus:
    --pixel_size_render  um/pixel for the output canvas (default 0.0645)
    --frame_count        number of GIF frames to emit (default 131)
    --output             path to mask GIF (default out/sim.gif)
    --pickle             path to results pickle (default out/sim_results.pkl)
    --fps                gif frames per second (default 10)
    --reuse_pickle       if set, skip the sim and read --pickle
"""

import argparse
import math
import os
import pickle
import time

from PIL import Image, ImageDraw

from process_bigraph import Composite, gather_emitter_results
from multi_cell.experiments.runner import PYMUNK_CORE
from partaker_to_vivamunk import make_document


def run_sim(csv_path, pixel_size, frame_interval, max_cells, sim_time):
    doc_fn, env_size, _rate, n_cells = make_document(
        csv_path, pixel_size, frame_interval, max_cells=max_cells,
    )
    print(f"Built document with {n_cells} cells, env {env_size:.1f} um")
    document = doc_fn({'env_size': env_size})
    sim = Composite({'state': document}, core=PYMUNK_CORE)
    print(f"Running sim for {sim_time:.0f} s ...")
    t0 = time.time()
    sim.run(sim_time)
    elapsed = time.time() - t0
    results = gather_emitter_results(sim)[('emitter',)]
    print(f"Sim done in {elapsed:.1f} s; {len(results)} emit steps")
    return results, env_size


def pick_frames_by_time(results, frame_count, frame_interval):
    """Pick `frame_count` results uniformly spaced in sim-time, target step
    spacing = frame_interval seconds (i.e. matches the real frame cadence)."""
    if not results:
        return []
    times = [r.get('time', i * frame_interval) for i, r in enumerate(results)]
    picks = []
    for k in range(frame_count):
        target = k * frame_interval
        idx = min(range(len(times)), key=lambda i: abs(times[i] - target))
        picks.append(results[idx])
    return picks


def draw_capsule(draw, cx_px, cy_px, length_px, radius_px, angle_rad,
                 fill=255):
    half_body = max(length_px * 0.5 - radius_px, 0.0)
    ux, uy = math.cos(angle_rad), math.sin(angle_rad)
    e1x, e1y = cx_px + ux * half_body, cy_px + uy * half_body
    e2x, e2y = cx_px - ux * half_body, cy_px - uy * half_body
    perp_x, perp_y = -uy * radius_px, ux * radius_px
    poly = [
        (e1x + perp_x, e1y + perp_y),
        (e1x - perp_x, e1y - perp_y),
        (e2x - perp_x, e2y - perp_y),
        (e2x + perp_x, e2y + perp_y),
    ]
    draw.polygon(poly, fill=fill)
    draw.ellipse([e1x - radius_px, e1y - radius_px,
                  e1x + radius_px, e1y + radius_px], fill=fill)
    draw.ellipse([e2x - radius_px, e2y - radius_px,
                  e2x + radius_px, e2y + radius_px], fill=fill)


def render_frame(state, env_size_um, pixel_size_um):
    canvas_px = int(round(env_size_um / pixel_size_um))
    img = Image.new('L', (canvas_px, canvas_px), 0)
    draw = ImageDraw.Draw(img)
    agents = state.get('agents', {}) or state.get('cells', {})
    for cell in agents.values():
        loc = cell.get('location') or (0.0, 0.0)
        cx_um, cy_um = float(loc[0]), float(loc[1])
        length_um = float(cell.get('length', 1.0))
        radius_um = float(cell.get('radius', 0.3))
        angle_rad = float(cell.get('angle', 0.0))
        if not all(map(math.isfinite,
                       [cx_um, cy_um, length_um, radius_um, angle_rad])):
            continue
        cx_px = cx_um / pixel_size_um
        cy_px = cy_um / pixel_size_um
        draw_capsule(
            draw, cx_px, cy_px,
            length_um / pixel_size_um,
            radius_um / pixel_size_um,
            angle_rad,
        )
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Render the Partaker-seeded smoke sim as a binary-mask GIF')
    parser.add_argument('csv', nargs='?',
                        default='/Volumes/SAM1/server_workspace_backup/'
                                'cell_history_amby.csv')
    parser.add_argument('--pixel_size', type=float, default=0.0645)
    parser.add_argument('--frame_interval', type=float, default=300)
    parser.add_argument('--sim_time', type=float, default=39000)
    parser.add_argument('--max_cells', type=int, default=50)
    parser.add_argument('--pixel_size_render', type=float, default=0.0645)
    parser.add_argument('--frame_count', type=int, default=131)
    parser.add_argument('--output', default='out/sim.gif')
    parser.add_argument('--pickle', default='out/sim_results.pkl')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--reuse_pickle', action='store_true')
    args = parser.parse_args()

    if args.reuse_pickle and os.path.exists(args.pickle):
        print(f"Loading cached results from {args.pickle}")
        with open(args.pickle, 'rb') as f:
            payload = pickle.load(f)
        results, env_size = payload['results'], payload['env_size']
        print(f"Loaded {len(results)} steps; env_size {env_size:.1f} um")
    else:
        results, env_size = run_sim(
            args.csv, args.pixel_size, args.frame_interval,
            args.max_cells, args.sim_time,
        )
        os.makedirs(os.path.dirname(args.pickle) or '.', exist_ok=True)
        with open(args.pickle, 'wb') as f:
            pickle.dump({'results': results, 'env_size': env_size,
                         'sim_time': args.sim_time,
                         'max_cells': args.max_cells}, f)
        print(f"Pickled timeseries -> {args.pickle}")

    picks = pick_frames_by_time(results, args.frame_count, args.frame_interval)
    print(f"Selected {len(picks)} frames at {args.frame_interval}s spacing")

    pil_frames = []
    for i, state in enumerate(picks):
        img = render_frame(state, env_size, args.pixel_size_render)
        pil_frames.append(img.convert('P'))
        if i % 10 == 0:
            print(f"  rendered frame {i}/{len(picks)-1}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
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
          f"({len(pil_frames)} frames @ {args.fps} fps, "
          f"{pil_frames[0].size[0]}x{pil_frames[0].size[1]})")


if __name__ == '__main__':
    main()
