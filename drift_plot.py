"""
drift_plot.py
-------------
Publication-ready divergence plot: sim vs real, frames 0-27 (pre-IPTG window).

Two stacked panels, log-scale, normalized to t=0:
    A) cell count   N(t) / N(0)
    B) total area   A(t) / A(0)

Real data: enhanced_tracking_data_all_tracks.csv (per cell, per frame).
Sim data:  re-runs the same Composite as sim_mask_renderer.py and reads
           emitter history. Uses out/sim_results.pkl if present.

Usage:
    python drift_plot.py \
        --real_csv /Volumes/SAM1/server_workspace_backup/enhanced_tracking_data_all_tracks.csv \
        --sim_csv  /Volumes/SAM1/server_workspace_backup/cell_history_amby.csv \
        --out      out/divergence.png
"""

import argparse
import csv
import math
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

from process_bigraph import Composite, gather_emitter_results
from multi_cell.experiments.runner import PYMUNK_CORE
from partaker_to_vivamunk import make_document


# ── style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

REAL_COLOR  = '#1f4e79'
SIM_COLOR   = '#c4393a'
HYDRO_COLOR = '#2ca02c'
GUIDE_COLOR = '#888888'

PRE_IPTG_LAST_FRAME = 27
FRAME_INTERVAL_MIN  = 5
PRE_IPTG_END_MIN    = PRE_IPTG_LAST_FRAME * FRAME_INTERVAL_MIN  # 135


# ── real data ────────────────────────────────────────────────────────

def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _frame_value(row):
    for k in ('frame', 'frame_idx', 'time_point', 'time', 'time_index', 't'):
        if k in row:
            v = _to_float(row[k])
            if v is not None:
                return int(v)
    return None


def _length_width_um(row, pixel_size):
    """Try common column name pairs; returns (length_um, width_um) or None."""
    pairs = (('length', 'width'),
             ('cell_length', 'cell_width'),
             ('major_axis', 'minor_axis'),
             ('major_axis_length', 'minor_axis_length'))
    for kl, kw in pairs:
        if kl in row and kw in row:
            L = _to_float(row[kl])
            W = _to_float(row[kw])
            if L is None or W is None or L <= 0 or W <= 0:
                continue
            return L * pixel_size, W * pixel_size
    return None


def real_trajectory(csv_path, pixel_size=0.0645, max_frame=PRE_IPTG_LAST_FRAME):
    counts = {}
    areas  = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = _frame_value(row)
            if frame is None or frame < 0 or frame > max_frame:
                continue
            lw = _length_width_um(row, pixel_size)
            if lw is None:
                continue
            L_um, W_um = lw
            r = W_um / 2.0
            area = max(L_um - 2*r, 0.0) * W_um + math.pi * r * r
            counts[frame] = counts.get(frame, 0) + 1
            areas[frame]  = areas.get(frame, 0.0) + area

    if not counts:
        raise ValueError(
            f"No usable rows found in {csv_path}. "
            f"Tried frame columns (frame/frame_idx/time) and length/width pairs.")

    frames = sorted(counts.keys())
    t_min  = np.array([f * FRAME_INTERVAL_MIN for f in frames])
    n_arr  = np.array([counts[f] for f in frames], dtype=float)
    a_arr  = np.array([areas[f]  for f in frames], dtype=float)
    return t_min, n_arr, a_arr


# ── sim data ─────────────────────────────────────────────────────────

def _capsule_area_um(length_um, radius_um):
    return max(length_um - 2*radius_um, 0.0) * (2*radius_um) + math.pi * radius_um**2


def run_sim_for_trajectory(csv_path, pixel_size, frame_interval,
                           max_cells, sim_time, hydro=False,
                           hydro_velocity_csv='', hydro_pressure_csv='',
                           hydro_negate_vx=False, hydro_negate_vy=False):
    doc_fn, env_size, _rate, n_cells = make_document(
        csv_path, pixel_size, frame_interval, max_cells=max_cells,
        hydro=hydro,
        hydro_velocity_csv=hydro_velocity_csv,
        hydro_pressure_csv=hydro_pressure_csv,
        hydro_negate_vx=hydro_negate_vx,
        hydro_negate_vy=hydro_negate_vy,
    )
    document = doc_fn({'env_size': env_size})
    sim = Composite({'state': document}, core=PYMUNK_CORE)
    t0 = time.time()
    sim.run(sim_time)
    elapsed = time.time() - t0
    results = gather_emitter_results(sim)[('emitter',)]
    label = 'with hydro' if hydro else 'defaults'
    print(f"  sim ({label}): {len(results)} emit steps in {elapsed:.1f} s "
          f"(seed cells: {n_cells})")
    return results, env_size


def load_or_run_sim(pickle_path, hydro, args):
    if pickle_path and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            payload = pickle.load(f)
        print(f"  reused pickle: {pickle_path} ({len(payload['results'])} steps)")
        return payload['results']
    results, env_size = run_sim_for_trajectory(
        args.sim_csv, args.pixel_size, args.frame_interval,
        args.max_cells, args.sim_time, hydro=hydro,
        hydro_velocity_csv=args.hydro_velocity_csv if hydro else '',
        hydro_pressure_csv=args.hydro_pressure_csv if hydro else '',
        hydro_negate_vx=args.hydro_negate_vx if hydro else False,
        hydro_negate_vy=args.hydro_negate_vy if hydro else False,
    )
    if pickle_path:
        os.makedirs(os.path.dirname(pickle_path) or '.', exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump({'results': results, 'env_size': env_size,
                         'sim_time': args.sim_time,
                         'max_cells': args.max_cells, 'hydro': hydro}, f)
        print(f"  pickled: {pickle_path}")
    return results


def sim_trajectory(results, max_time_s=PRE_IPTG_END_MIN * 60):
    """Returns (t_min, count, total_area_um2) clipped to pre-IPTG window."""
    t_min = []
    counts = []
    areas  = []
    for i, r in enumerate(results):
        t_s = float(r.get('time', i * 30.0))
        if t_s > max_time_s:
            break
        agents = r.get('agents') or r.get('cells') or {}
        if not isinstance(agents, dict):
            continue
        n = 0
        a_total = 0.0
        for cell in agents.values():
            L = float(cell.get('length', 0.0))
            R = float(cell.get('radius', 0.0))
            if not (math.isfinite(L) and math.isfinite(R)) or L <= 0 or R <= 0:
                continue
            n += 1
            a_total += _capsule_area_um(L, R)
        t_min.append(t_s / 60.0)
        counts.append(n)
        areas.append(a_total)
    return np.array(t_min), np.array(counts, dtype=float), np.array(areas, dtype=float)


# ── plot ─────────────────────────────────────────────────────────────

def make_plot(t_real, n_real, a_real,
              t_sim, n_sim, a_sim,
              t_hyd, n_hyd, a_hyd,
              out_path):
    if n_real[0] == 0 or n_sim[0] == 0 or n_hyd[0] == 0:
        raise ValueError("First-timepoint count is zero; cannot normalize.")

    n_real_n = n_real / n_real[0]
    a_real_n = a_real / a_real[0]
    n_sim_n  = n_sim  / n_sim[0]
    a_sim_n  = a_sim  / a_sim[0]
    n_hyd_n  = n_hyd  / n_hyd[0]
    a_hyd_n  = a_hyd  / a_hyd[0]

    fig, (ax_n, ax_a) = plt.subplots(
        2, 1, figsize=(7.2, 7.8), sharex=True,
        gridspec_kw={'hspace': 0.18},
    )

    for ax, real_n, sim_n, hyd_n, ylabel, title in (
        (ax_n, n_real_n, n_sim_n, n_hyd_n,
         r'Cell count   $N(t)/N(0)$', 'Population growth'),
        (ax_a, a_real_n, a_sim_n, a_hyd_n,
         r'Total area   $A(t)/A(0)$', 'Biomass growth'),
    ):
        ax.plot(t_sim, sim_n, '-', color=SIM_COLOR, lw=2.0,
                label='Sim (defaults)')
        ax.plot(t_hyd, hyd_n, '-', color=HYDRO_COLOR, lw=2.0,
                label='Sim + hydrodynamics')
        ax.plot(t_real, real_n, 'o-', color=REAL_COLOR, lw=1.6, ms=4.5,
                label='Real (Partaker)')
        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontweight='bold', pad=6)
        ax.grid(True, which='major', linestyle=':', alpha=0.45)
        ax.grid(True, which='minor', linestyle=':', alpha=0.18)
        ax.axvline(PRE_IPTG_END_MIN, color=GUIDE_COLOR,
                   linestyle='--', linewidth=1.0)

    # IPTG annotation only on top panel (cleaner)
    y_top = ax_n.get_ylim()[1]
    ax_n.text(PRE_IPTG_END_MIN - 1, y_top * 0.92,
              'IPTG / out of scope', rotation=90,
              ha='right', va='top', color=GUIDE_COLOR, fontsize=9)

    ax_n.legend(loc='upper left')
    ax_a.set_xlabel('Time (min)')
    ax_a.set_xlim(0, PRE_IPTG_END_MIN + 2)

    fig.suptitle(
        'Sim vs. real divergence — pre-IPTG window (frames 0–27)',
        fontweight='bold', y=0.995,
    )

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(os.path.splitext(out_path)[0] + '.pdf')
    plt.close(fig)
    print(f"  wrote: {out_path}")
    print(f"  wrote: {os.path.splitext(out_path)[0] + '.pdf'}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real_csv', required=True,
                   help='Path to enhanced_tracking_data_all_tracks.csv')
    p.add_argument('--sim_csv',  required=True,
                   help='Path to cell_history_*.csv (used to seed sim)')
    p.add_argument('--pixel_size', type=float, default=0.0645)
    p.add_argument('--frame_interval', type=float, default=300)
    p.add_argument('--max_cells', type=int, default=50)
    p.add_argument('--sim_time', type=float, default=PRE_IPTG_END_MIN * 60)
    p.add_argument('--out', default='out/divergence.png')
    p.add_argument('--sim_pickle', default='out/sim_default.pkl',
                   help='Cache for the framework-defaults sim.')
    p.add_argument('--sim_pickle_hydro', default='out/sim_hydro.pkl',
                   help='Cache for defaults + hydrodynamics sim.')
    p.add_argument('--hydro_velocity_csv', default='',
                   help='COMSOL velocity CSV; if set, hydro sim uses real chamber flow.')
    p.add_argument('--hydro_pressure_csv', default='',
                   help='COMSOL pressure CSV (pairs with --hydro_velocity_csv).')
    p.add_argument('--hydro_negate_vx', action='store_true',
                   help='Flip x-component of COMSOL flow direction.')
    p.add_argument('--hydro_negate_vy', action='store_true',
                   help='Flip y-component of COMSOL flow direction.')
    args = p.parse_args()

    print('Real data:')
    t_real, n_real, a_real = real_trajectory(
        args.real_csv, pixel_size=args.pixel_size,
        max_frame=PRE_IPTG_LAST_FRAME,
    )
    print(f"  frames: {len(t_real)}   "
          f"count {int(n_real[0])} -> {int(n_real[-1])}   "
          f"area {a_real[0]:.1f} -> {a_real[-1]:.1f} um^2")

    print('Sim (framework defaults):')
    results_def = load_or_run_sim(args.sim_pickle, hydro=False, args=args)
    t_sim, n_sim, a_sim = sim_trajectory(results_def, max_time_s=args.sim_time)
    print(f"  steps:  {len(t_sim)}   "
          f"count {int(n_sim[0])} -> {int(n_sim[-1])}   "
          f"area {a_sim[0]:.1f} -> {a_sim[-1]:.1f} um^2")

    print('Sim (defaults + hydrodynamics):')
    results_hyd = load_or_run_sim(args.sim_pickle_hydro, hydro=True, args=args)
    t_hyd, n_hyd, a_hyd = sim_trajectory(results_hyd, max_time_s=args.sim_time)
    print(f"  steps:  {len(t_hyd)}   "
          f"count {int(n_hyd[0])} -> {int(n_hyd[-1])}   "
          f"area {a_hyd[0]:.1f} -> {a_hyd[-1]:.1f} um^2")

    print('Plotting:')
    make_plot(t_real, n_real, a_real,
              t_sim, n_sim, a_sim,
              t_hyd, n_hyd, a_hyd,
              args.out)


if __name__ == '__main__':
    main()
