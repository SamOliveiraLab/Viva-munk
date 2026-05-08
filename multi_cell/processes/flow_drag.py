"""
FlowDrag — low-Reynolds Stokes-drag flow on agents.

At bacterial scale (Re ~ 1e-5) viscous drag dominates inertia, so cells move
with the surrounding fluid almost instantly. In that limit v_cell ≈ v_fluid,
so each step we translate the agent location by v_fluid(x, y) * dt.

Two modes for the velocity field:

  * mode='analytical'  — linear shear gradient along one axis. Useful as a
                         placeholder before chip-specific data is available.
  * mode='comsol'      — read a chamber-specific COMSOL flow field from CSVs.
                         The exporter we work with provides:
                            cell_id, x, y, z, velocity_m_s
                            cell_id, x, y, z, pressure_Pa
                         Velocity is exported as scalar magnitude |v| only;
                         the flow direction is reconstructed from the
                         pressure gradient (Stokes flow: v ∝ −∇p). At each
                         sim cell, |v| is interpolated from the nearest
                         COMSOL samples (inverse-distance weighting), and
                         ∇p is fit by least-squares through the k nearest
                         pressure samples. Combined: v(x,y) = |v| * (−∇p / |∇p|).
"""
import csv
import math

from process_bigraph import Step

try:
    import numpy as np
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    np = None
    cKDTree = None


class FlowDrag(Step):
    config_schema = {
        'mode':            {'_type': 'string', '_default': 'analytical'},
        # analytical-mode params
        'v_max':           {'_type': 'float',  '_default': 0.1},
        'axis_max':        {'_type': 'float',  '_default': 70.0},
        'axis':            {'_type': 'string', '_default': 'y'},
        # comsol-mode params
        'velocity_csv':    {'_type': 'string', '_default': ''},
        'pressure_csv':    {'_type': 'string', '_default': ''},
        'k_neighbors':     {'_type': 'integer', '_default': 8},
        'velocity_units':  {'_type': 'string', '_default': 'm/s'},  # or 'um/s'
        # shared
        'interval':        {'_type': 'float',  '_default': 30.0},
        'viscosity':       {'_type': 'float',  '_default': 6.91e-4},
        'agents_key':      {'_type': 'string', '_default': 'cells'},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self.mode = self.config['mode']
        self.dt   = float(self.config['interval'])
        if self.mode == 'comsol':
            if np is None or cKDTree is None:
                raise ImportError(
                    "FlowDrag(mode='comsol') requires numpy + scipy.spatial.cKDTree")
            self._load_comsol()
        elif self.mode == 'analytical':
            self.v_max    = float(self.config['v_max'])
            self.axis_max = float(self.config['axis_max'])
            self.axis     = self.config['axis']
            if self.axis not in ('x', 'y'):
                raise ValueError(f"axis must be 'x' or 'y', got {self.axis!r}")
        else:
            raise ValueError(f"unknown mode {self.mode!r}")

    # ── COMSOL data loader ──────────────────────────────────────────

    @staticmethod
    def _read_xyv(path, value_col):
        xs, ys, vs = [], [], []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x = float(row['x']); y = float(row['y'])
                    v = float(row[value_col])
                except (KeyError, ValueError):
                    continue
                xs.append(x); ys.append(y); vs.append(v)
        return np.column_stack([xs, ys]), np.asarray(vs)

    def _load_comsol(self):
        v_xy, v_mag = self._read_xyv(self.config['velocity_csv'], 'velocity_m_s')
        p_xy, p_val = self._read_xyv(self.config['pressure_csv'], 'pressure_Pa')
        # convert velocity to um/s if needed
        if self.config['velocity_units'] == 'm/s':
            v_mag = v_mag * 1e6
        self._v_xy  = v_xy
        self._v_mag = v_mag.astype(float)
        self._v_tree = cKDTree(v_xy)
        self._p_xy  = p_xy
        self._p_val = p_val.astype(float)
        self._p_tree = cKDTree(p_xy)
        self._k = int(self.config['k_neighbors'])
        print(f"FlowDrag(comsol): velocity samples {len(v_xy)}  "
              f"|v| range {v_mag.min():.3g}–{v_mag.max():.3g} um/s   "
              f"pressure samples {len(p_xy)}  "
              f"p range {p_val.min():.3f}–{p_val.max():.3f} Pa")

    # ── velocity-field lookups ──────────────────────────────────────

    def _v_fluid_analytical(self, x, y):
        if self.axis == 'y':
            frac = max(0.0, min(1.0, y / self.axis_max if self.axis_max > 0 else 0.0))
            return 0.0, self.v_max * frac
        frac = max(0.0, min(1.0, x / self.axis_max if self.axis_max > 0 else 0.0))
        return self.v_max * frac, 0.0

    def _v_fluid_comsol(self, x, y):
        # |v| from k-nearest velocity samples (inverse-distance weighting)
        dists, idxs = self._v_tree.query([x, y], k=self._k)
        if dists[0] < 1e-9:
            v_mag = float(self._v_mag[idxs[0]])
        else:
            w = 1.0 / dists
            v_mag = float(np.sum(w * self._v_mag[idxs]) / np.sum(w))

        # ∇p from least-squares plane fit through k-nearest pressure samples
        _, p_idx = self._p_tree.query([x, y], k=self._k)
        Xs = self._p_xy[p_idx, 0]
        Ys = self._p_xy[p_idx, 1]
        Ps = self._p_val[p_idx]
        A = np.column_stack([Xs, Ys, np.ones_like(Xs)])
        try:
            coef, *_ = np.linalg.lstsq(A, Ps, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0
        gx, gy = float(coef[0]), float(coef[1])
        gmag = math.sqrt(gx * gx + gy * gy)
        if gmag < 1e-12:
            return 0.0, 0.0
        # Stokes / low-Re: flow points down the pressure gradient
        return v_mag * (-gx / gmag), v_mag * (-gy / gmag)

    # ── Step interface ──────────────────────────────────────────────

    def inputs(self):
        return {'agents': 'map[pymunk_agent]'}

    def outputs(self):
        return {'agents': 'map[pymunk_agent]'}

    def update(self, state):
        agents = state.get('agents', {}) or {}
        updates = {}
        for aid, agent in agents.items():
            loc = agent.get('location')
            if loc is None:
                continue
            x, y = float(loc[0]), float(loc[1])
            if self.mode == 'comsol':
                vx, vy = self._v_fluid_comsol(x, y)
            else:
                vx, vy = self._v_fluid_analytical(x, y)
            updates[aid] = {'location': (x + vx * self.dt, y + vy * self.dt)}
        return {'agents': updates}


def make_flow_drag_process(
    mode='analytical',
    v_max=0.1, axis_max=70.0, axis='y',
    velocity_csv='', pressure_csv='', k_neighbors=8, velocity_units='m/s',
    interval=30.0, viscosity=6.91e-4, agents_key='cells',
):
    return {
        '_type': 'step',
        'address': 'local:FlowDrag',
        'config': {
            'mode': mode,
            'v_max': v_max,
            'axis_max': axis_max,
            'axis': axis,
            'velocity_csv': velocity_csv,
            'pressure_csv': pressure_csv,
            'k_neighbors': k_neighbors,
            'velocity_units': velocity_units,
            'interval': interval,
            'viscosity': viscosity,
            'agents_key': agents_key,
        },
        'inputs':  {'agents': [agents_key]},
        'outputs': {'agents': [agents_key]},
    }
