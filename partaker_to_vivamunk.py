"""
partaker_to_vivamunk.py
-----------------------
Initializes a Viva-munk Daughter Machine with real Partaker cell data.

Place this file in the Viva-munk repo root, then run:

    python partaker_to_vivamunk.py path/to/cell_history_p0.csv

Optional flags:
    --pixel_size      um/pixel  (default 0.0645 for 100x Nikon confocal)
    --frame_interval  seconds between frames (default 300 = 5 min)
    --sim_time        simulation seconds (default 28800 = 8 h)
    --max_cells       cap loaded cells from frame 0 for smoke tests (default: no cap)
"""

import argparse
import csv
import math
import os

from process_bigraph.emitter import emitter_from_wires
from multi_cell.processes.grow_divide import add_grow_divide_to_agents
from multi_cell.processes.remove_crossing import make_remove_crossing_process
from multi_cell.experiments.runner import run_experiment


# ── helpers ──────────────────────────────────────────────────────────

def parse_array(s):
    if not s or s.strip() == '':
        return []
    return [float(x.strip()) for x in s.split(',')]


def capsule_mass(length, radius, density=0.02):
    cyl_area = length * 2.0 * radius
    cap_area = math.pi * radius ** 2
    return density * (cyl_area + cap_area)


# ── load ─────────────────────────────────────────────────────────────

def load_partaker_cells(csv_path, pixel_size):
    cells = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['start_time']) != 0:
                continue

            lengths = parse_array(row['lengths'])
            widths  = parse_array(row['widths'])
            orientations = parse_array(row['orientations'])
            x_pos = parse_array(row['x_positions'])
            y_pos = parse_array(row['y_positions'])
            morphs = row['morphology_classes'].split(',')

            if not lengths or not x_pos:
                continue
            if morphs[0].strip() == 'Artifact':
                continue

            length_um = lengths[0] * pixel_size
            width_um  = widths[0] * pixel_size

            # Partaker occasionally emits NaN for length/width/angle/position;
            # range filters below silently let NaN through (nan < x is False),
            # which becomes a fatal mass=NaN inside Chipmunk.
            vals = [length_um, width_um, orientations[0], x_pos[0], y_pos[0]]
            if any(not math.isfinite(v) for v in vals):
                continue

            # keep only biologically plausible single E. coli cells
            if length_um < 0.5 or length_um > 15.0:
                continue
            if width_um < 0.3 or width_um > 3.0:
                continue

            cells.append({
                'cell_id':  row['cell_id'],
                'x':        x_pos[0] * pixel_size,
                'y':        y_pos[0] * pixel_size,
                'length':   length_um,
                'width':    width_um,
                'angle':    orientations[0],
                'lifespan': int(row['lifespan']),
                'fate':     row['fate'],
            })
    return cells


def estimate_growth_rate(cells, frame_interval):
    dts = [c['lifespan'] * frame_interval
           for c in cells if c['fate'] == 'divided' and c['lifespan'] > 0]
    if dts:
        avg = sum(dts) / len(dts)
        rate = math.log(2) / avg
    else:
        avg = 2400.0
        rate = math.log(2) / avg
    return rate, avg, len(dts)


def build_agents(cells, env_size):
    density = 0.02
    agents = {}
    for c in cells:
        aid = f"p{c['cell_id']}"
        radius = max(c['width'] / 2.0, 0.3)
        length = max(c['length'], radius * 2.0)
        mass = capsule_mass(length, radius, density)

        pad = radius + 1.0
        x = max(pad, min(c['x'], env_size - pad))
        y = max(pad, min(c['y'], env_size - pad))

        agents[aid] = {
            'id':         aid,
            'type':       'segment',
            'mass':       float(mass),
            'length':     float(length),
            'radius':     float(radius),
            'angle':      float(c['angle']),
            'location':   (float(x), float(y)),
            'velocity':   (0.0, 0.0),
            'elasticity': 0.1,
            'adhesins':   1.0,
        }
    return agents


# ── document builder ─────────────────────────────────────────────────

def make_document(csv_path, pixel_size, frame_interval, max_cells=None):
    """Returns (document_fn, env_size, growth_rate, n_cells)."""

    cells = load_partaker_cells(csv_path, pixel_size)
    print(f"Loaded {len(cells)} valid cells from frame 0")
    if max_cells is not None and len(cells) > max_cells:
        cells = cells[:max_cells]
        print(f"Capped to first {len(cells)} cells (--max_cells)")

    all_x = [c['x'] for c in cells]
    all_y = [c['y'] for c in cells]
    env_size = max(max(all_x), max(all_y)) + 10.0
    env_size = max(env_size, 30.0)
    print(f"Environment size: {env_size:.1f} um")

    agents = build_agents(cells, env_size)
    print(f"Created {len(agents)} agents")

    # Rate + mutate left at framework defaults. Threshold scaled to µm cell
    # size — unit conversion so the framework operates at our scale.
    rate = 0.000289
    sample = list(agents.values())[0]
    division_threshold = 0.02 * (2 * sample['radius']) * (sample['length'] * 2.0)
    print(f"Growth params: rate={rate} /s (default ~40-min doubling), "
          f"threshold={division_threshold:.4f} (scaled to µm cell size), mutate=False")

    initial_state = {'cells': agents, 'particles': {}}
    add_grow_divide_to_agents(
        initial_state,
        agents_key='cells',
        config={
            'agents_key': 'cells',
            'threshold': division_threshold,
        },
    )

    interval = 30.0
    flow_x = env_size * 0.95
    n_cells = len(agents)

    def document_fn(config=None):
        return {
            'cells':     initial_state['cells'],
            'particles': initial_state['particles'],
            'multibody': {
                '_type': 'process',
                'address': 'local:PymunkProcess',
                'config': {
                    'env_size': env_size,
                    'elasticity': 0.1,
                },
                'interval': interval,
                'inputs': {
                    'segment_cells': ['cells'],
                    'circle_particles': ['particles'],
                },
                'outputs': {
                    'segment_cells': ['cells'],
                    'circle_particles': ['particles'],
                },
            },
            'remove_crossing': make_remove_crossing_process(
                x_max=flow_x,
                agents_key='cells',
            ),
            'emitter': emitter_from_wires({
                'agents': ['cells'],
                'particles': ['particles'],
                'time': ['global_time'],
            }),
        }

    return document_fn, env_size, rate, n_cells


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run Viva-munk Daughter Machine with real Partaker data')
    parser.add_argument('csv', help='Path to cell_history_p0.csv')
    parser.add_argument('--pixel_size', type=float, default=0.0645,
                        help='um/pixel (default: 0.0645 for 100x)')
    parser.add_argument('--frame_interval', type=float, default=300,
                        help='Seconds between frames (default: 300 = 5 min)')
    parser.add_argument('--sim_time', type=float, default=28800,
                        help='Simulation seconds (default: 28800 = 8h)')
    parser.add_argument('--max_cells', type=int, default=None,
                        help='Cap number of frame-0 cells loaded (smoke tests)')
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  PARTAKER -> VIVA-MUNK DAUGHTER MACHINE")
    print(f"  Pixel size:     {args.pixel_size} um/px")
    print(f"  Frame interval: {args.frame_interval} s")
    print(f"  Sim time:       {args.sim_time} s ({args.sim_time/3600:.1f} h)")
    if args.max_cells is not None:
        print(f"  Max cells:      {args.max_cells}")
    print(f"{'='*50}\n")

    doc_fn, env_size, rate, n_cells = make_document(
        args.csv, args.pixel_size, args.frame_interval,
        max_cells=args.max_cells,
    )

    entry = {
        'document': doc_fn,
        'time': args.sim_time,
        'config': {'env_size': env_size},
        'description': f'Daughter machine initialized with {n_cells} Partaker cells',
    }

    result = run_experiment(
        'partaker_daughter_machine',
        output_dir='out',
        entry=entry,
    )

    print(f"\nDone! Output in out/")
    print(f"GIF: {result.get('gif_path', 'N/A')}")


if __name__ == '__main__':
    main()
