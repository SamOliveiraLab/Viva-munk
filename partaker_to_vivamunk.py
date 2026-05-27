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
from multi_cell.processes.flow_drag import make_flow_drag_process
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

def make_document(csv_path, pixel_size, frame_interval, max_cells=None,
                  hydro=False, hydro_velocity_csv='', hydro_pressure_csv='',
                  hydro_negate_vx=False, hydro_negate_vy=False,
                  attach=False,
                  chamber_length=None, chamber_width=None):
    """Returns (document_fn, env_size, growth_rate, n_cells)."""

    cells = load_partaker_cells(csv_path, pixel_size)
    print(f"Loaded {len(cells)} valid cells from frame 0")
    if max_cells is not None and len(cells) > max_cells:
        cells = cells[:max_cells]
        print(f"Capped to first {len(cells)} cells (--max_cells)")

    all_x = [c['x'] for c in cells]
    all_y = [c['y'] for c in cells]

    # Chamber geometry: use real dimensions if provided, else infer from data.
    # PymunkProcess uses a square domain (env_size x env_size), so we take the
    # larger dimension. remove_crossing sets the outlet boundary separately.
    if chamber_length is not None or chamber_width is not None:
        cl = chamber_length or (max(all_x) + 10.0)
        cw = chamber_width or (max(all_y) + 10.0)
        env_size = max(cl, cw)
        print(f"Chamber geometry: {cl:.1f} x {cw:.1f} um (from args)")
    else:
        env_size = max(max(all_x), max(all_y)) + 10.0
        print(f"Chamber geometry: inferred from cell coords")
    env_size = max(env_size, 30.0)
    print(f"Environment size: {env_size:.1f} um (square domain)")

    agents = build_agents(cells, env_size)
    print(f"Created {len(agents)} agents")

    # Growth rate measured from real division times in the tracking CSV.
    # Falls back to framework default (40-min doubling) if no divisions found.
    rate, avg_dt, n_div = estimate_growth_rate(cells, frame_interval)
    if n_div > 0:
        print(f"Growth rate: {rate:.6f} /s measured from {n_div} real divisions "
              f"(avg doubling {avg_dt:.0f} s = {avg_dt/60:.1f} min)")
    else:
        print(f"Growth rate: {rate:.6f} /s (framework default, no divisions in CSV)")
    sample = list(agents.values())[0]
    division_threshold = 0.02 * (2 * sample['radius']) * (sample['length'] * 2.0)
    print(f"Division threshold: {division_threshold:.4f} (scaled to um cell size)")

    initial_state = {'cells': agents, 'particles': {}}
    add_grow_divide_to_agents(
        initial_state,
        agents_key='cells',
        config={
            'agents_key': 'cells',
            'rate': rate,
            'threshold': division_threshold,
        },
    )

    interval = 30.0
    flow_x = env_size * 0.95
    n_cells = len(agents)
    # Stage-3 environment rule: real low-Reynolds Stokes-drag flow field.
    # At bacterial scale (Re ~ 1e-5), v_cell ≈ v_fluid, so cells are advected
    # by a parameterised flow profile. Linear shear along x from closed wall
    # (x=0) to the outlet at flow_x — same axis as the existing remove_crossing
    # boundary, so flow drives cells toward the outlet rather than into a wall.
    # Analytical first-pass profile; COMSOL flow field swaps in here later.
    hydro_v_max = 0.05  # um/s at outlet
    hydro_shear_rate = hydro_v_max / flow_x  # 1/s, ∂v_x/∂x
    if hydro:
        print(f"Hydrodynamics rule: ON  v_x(x) = {hydro_v_max} um/s × (x / "
              f"{flow_x:.1f}), shear ∂v/∂x = {hydro_shear_rate:.4g} 1/s, "
              f"low-Re Stokes-drag limit (mu_water ≈ 6.91e-4 Pa·s)")

    if attach:
        print(f"Attachment rule: ON  surface=bottom, threshold=0.5, "
              f"distance={env_size:.1f} um (full monolayer, all cells start attached)")

    def document_fn(config=None):
        multibody_config = {
            'env_size': env_size,
            'elasticity': 0.1,
        }
        if attach:
            # Monolayer chip: all cells sit on the glass surface.
            # adhesion_distance = env_size ensures every cell in the domain
            # is considered "touching" the surface, matching the real chip
            # where chamber height ~ cell diameter (all cells on glass).
            multibody_config.update({
                'adhesion_enabled': True,
                'adhesion_surface': 'bottom',
                'adhesion_threshold': 0.5,
                'adhesion_distance': env_size,
            })
        doc = {
            'cells':     initial_state['cells'],
            'particles': initial_state['particles'],
            'multibody': {
                '_type': 'process',
                'address': 'local:PymunkProcess',
                'config': multibody_config,
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
        if hydro:
            if hydro_velocity_csv and hydro_pressure_csv:
                doc['flow_drag'] = make_flow_drag_process(
                    mode='comsol',
                    velocity_csv=hydro_velocity_csv,
                    pressure_csv=hydro_pressure_csv,
                    negate_vx=hydro_negate_vx,
                    negate_vy=hydro_negate_vy,
                    interval=interval,
                    agents_key='cells',
                )
            else:
                doc['flow_drag'] = make_flow_drag_process(
                    mode='analytical',
                    v_max=hydro_v_max,
                    axis_max=flow_x,
                    axis='x',
                    interval=interval,
                    agents_key='cells',
                )
        return doc

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
    parser.add_argument('--hydro', action='store_true',
                        help='Enable hydrodynamics rule (Stokes-drag flow on cells)')
    parser.add_argument('--attach', action='store_true',
                        help='Enable surface attachment (cells pin to glass via adhesins)')
    parser.add_argument('--chamber_length', type=float, default=None,
                        help='Real chamber length in um (x-axis). Overrides auto-computed '
                             'env_size. Get from COMSOL model or chip design.')
    parser.add_argument('--chamber_width', type=float, default=None,
                        help='Real chamber width in um (y-axis). Overrides auto-computed '
                             'env_size. Get from COMSOL model or chip design.')
    parser.add_argument('--hydro_velocity_csv', default='',
                        help='COMSOL velocity CSV (cell_id,x,y,z,velocity_m_s). '
                             'If set together with --hydro_pressure_csv, real '
                             'chamber flow is used; otherwise analytical shear.')
    parser.add_argument('--hydro_pressure_csv', default='',
                        help='COMSOL pressure CSV (cell_id,x,y,z,pressure_Pa). '
                             'Pressure gradient gives flow direction.')
    parser.add_argument('--hydro_negate_vx', action='store_true',
                        help='Flip x-component of COMSOL flow (use if COMSOL '
                             'chamber outlet is mirrored relative to sim).')
    parser.add_argument('--hydro_negate_vy', action='store_true',
                        help='Flip y-component of COMSOL flow.')
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
        hydro=args.hydro,
        hydro_velocity_csv=args.hydro_velocity_csv,
        hydro_pressure_csv=args.hydro_pressure_csv,
        hydro_negate_vx=args.hydro_negate_vx,
        hydro_negate_vy=args.hydro_negate_vy,
        attach=args.attach,
        chamber_length=args.chamber_length,
        chamber_width=args.chamber_width,
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
