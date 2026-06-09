"""
rule_time_grid.py
-----------------
The "time x rule" grid Sam asked for: top row = real microscopy, each row
below adds one rule, columns = frames. Same idea as rule_montage.py but a 2D
grid (rules down, time across) instead of a single-frame strip.

It reuses the repo's own renderers, so every cell is pixel-identical to the
montage (same colors, same cell-id labels, same ROI crop, same rectangular
chamber). It reads the SAME montage_*.pkl pickles rule_montage.py already
wrote, so with --reuse_pickles there is no re-simulation.

Usage (reusing the pickles from your last DT run):
    python rule_time_grid.py \
      --sim_csv /Volumes/SAM1/server_workspace_backup/cell_history_amby.csv \
      --h5 /Volumes/SAM1/.../segmentation_cache.h5 \
      --roi_mask /Volumes/SAM1/.../roi_mask.npy \
      --frames 0 9 18 27 \
      --chamber_length 70 --chamber_width 52.5 \
      --max_cells 250 --sim_time 8100 \
      --pickle_dir out/DT_digital_twin_calibration_20260528/pickles --reuse_pickles \
      --out out/DT_digital_twin_calibration_20260528/rule_time_grid.png
"""

import argparse
import os
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_mask_renderer import run_sim, pick_frames_by_time, render_frame
from real_movie_renderer import load_mask_frames, crop_to_roi_bbox

# Rows top -> bottom. Real first (the target), then each rule cumulatively.
# suffix matches rule_montage.py's pickle naming: montage_<suffix>.pkl
ROWS = [
    ("Real",         "target",       "amber", "real", None,                                              None),
    ("Defaults",     "messy",        None,    "sim",  dict(hydro=False, attach=False, pressure=False),    "default"),
    ("+ Flow",       "washes out",   None,    "sim",  dict(hydro=True,  attach=False, pressure=False),    "hydro"),
    ("+ Attachment", "overshoots",   None,    "sim",  dict(hydro=True,  attach=True,  pressure=False),    "attach"),
    ("+ Crowding",   "matches real", "teal",  "sim",  dict(hydro=True,  attach=True,  pressure=True),     "pressure"),
]

AMBER = (186, 117, 23)
TEAL = (29, 158, 117)
INK = (38, 38, 36)
MUTE = (120, 119, 112)
WHITE = (255, 255, 255)
TILE_BG = (0, 0, 0)

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def load_font(size, bold=False):
    paths = []
    try:
        import matplotlib.font_manager as fm
        try:
            paths.append(fm.findfont("DejaVu Sans Bold" if bold else "DejaVu Sans",
                                     fallback_to_default=False))
        except Exception:
            pass
    except Exception:
        pass
    suffix = "-Bold" if bold else ""
    paths += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % suffix,
        "/System/Library/Fonts/Supplemental/Arial%s.ttf" % (" Bold" if bold else ""),
        "/Library/Fonts/Arial%s.ttf" % (" Bold" if bold else ""),
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def tw(draw, s, font):
    try:
        return draw.textlength(s, font=font)
    except Exception:
        return font.getlength(s)


def fit_tile(img, cw, ch):
    im = img.copy()
    im.thumbnail((cw, ch), RESAMPLE)
    tile = Image.new("RGB", (cw, ch), TILE_BG)
    tile.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
    return tile


def get_sim_results(args, flags, suffix):
    pk = os.path.join(args.pickle_dir, "montage_%s.pkl" % suffix)
    if args.reuse_pickles and os.path.exists(pk):
        print("  reuse %s" % pk)
        with open(pk, "rb") as f:
            payload = pickle.load(f)
        return payload["results"], payload["env_size"]
    results, env_size = run_sim(
        args.sim_csv, args.pixel_size, args.frame_interval,
        args.max_cells, args.sim_time,
        chamber_length=args.chamber_length, chamber_width=args.chamber_width,
        **flags,
    )
    os.makedirs(args.pickle_dir, exist_ok=True)
    with open(pk, "wb") as f:
        pickle.dump({"results": results, "env_size": env_size}, f)
    print("  pickled %s" % pk)
    return results, env_size


def render_sim_row(args, flags, suffix):
    results, env_size = get_sim_results(args, flags, suffix)
    picks = pick_frames_by_time(results, max(args.frames) + 1, args.frame_interval)
    out = []
    for idx in args.frames:
        state = picks[min(idx, len(picks) - 1)]
        img = render_frame(state, env_size, args.pixel_size_render,
                           color=True, label_ids=not args.no_ids,
                           font_size=args.font_size,
                           env_height_um=args.chamber_width)
        out.append(img.convert("RGB"))
    return out


def render_real_row(args):
    roi = np.load(args.roi_mask) if args.roi_mask else None
    frames = load_mask_frames(args.h5, args.position, args.channel,
                              max(args.frames), roi_mask=roi, color=True,
                              label_ids=not args.no_ids, font_size=args.font_size)
    if roi is not None:
        frames = crop_to_roi_bbox(frames, roi)
    return [Image.fromarray(frames[min(idx, len(frames) - 1)], mode="RGB")
            for idx in args.frames]


def build_grid(panels_2d, out_path, frames, title, min_per_frame,
               cell_w, cell_h, row_gap, col_gap):
    nrows, ncols = len(ROWS), len(frames)
    left = 168
    top = 84 if title else 48
    colhead = 26
    cap = 64
    right = 28
    bottom = 18
    W = left + ncols * cell_w + (ncols - 1) * col_gap + right
    H = top + colhead + nrows * cell_h + (nrows - 1) * row_gap + cap + bottom

    canvas = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(canvas)
    f_title = load_font(26, bold=True)
    f_row = load_font(20, bold=True)
    f_sub = load_font(15)
    f_col = load_font(17, bold=True)
    f_cap = load_font(14)

    if title:
        d.text((left, 30), title, font=f_title, fill=INK)

    grid_top = top + colhead
    for c, idx in enumerate(frames):
        cx = left + c * (cell_w + col_gap) + cell_w // 2
        lab = "frame %d" % idx
        d.text((cx - tw(d, lab, f_col) / 2, top), lab, font=f_col, fill=INK)

    for r, (label, sub, accent, *_rest) in enumerate(ROWS):
        ry = grid_top + r * (cell_h + row_gap)
        lab_color = AMBER if accent == "amber" else INK
        sub_color = TEAL if accent == "teal" else (AMBER if accent == "amber" else MUTE)
        d.text((left - 14 - tw(d, label, f_row), ry + cell_h // 2 - 20),
               label, font=f_row, fill=lab_color)
        if sub:
            d.text((left - 14 - tw(d, sub, f_sub), ry + cell_h // 2 + 6),
                   sub, font=f_sub, fill=sub_color)
        for c in range(ncols):
            canvas.paste(fit_tile(panels_2d[r][c], cell_w, cell_h),
                         (left + c * (cell_w + col_gap), ry))

    cap_y = grid_top + nrows * cell_h + (nrows - 1) * row_gap + 14
    total_min = "%g" % (max(frames) * min_per_frame)
    d.text((left, cap_y),
           "Top row is real microscopy. Each row below adds one rule, cumulatively.",
           font=f_cap, fill=MUTE)
    d.text((left, cap_y + 20),
           "Columns are frames at %g min intervals (pre-IPTG window, 0 to %s min)."
           % (min_per_frame, total_min), font=f_cap, fill=MUTE)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, dpi=(200, 200))
    print("\nSaved %s  (%d x %d)" % (out_path, W, H))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_csv", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--roi_mask", default="")
    p.add_argument("--position", type=int, default=0)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 9, 18, 27])
    p.add_argument("--pixel_size", type=float, default=0.0645)
    p.add_argument("--pixel_size_render", type=float, default=0.0645)
    p.add_argument("--frame_interval", type=float, default=300)
    p.add_argument("--sim_time", type=float, default=8100)
    p.add_argument("--max_cells", type=int, default=250)
    p.add_argument("--chamber_length", type=float, default=None)
    p.add_argument("--chamber_width", type=float, default=None)
    p.add_argument("--font_size", type=int, default=9)
    p.add_argument("--no_ids", action="store_true")
    p.add_argument("--pickle_dir", default="out/montage_pickles")
    p.add_argument("--reuse_pickles", action="store_true")
    p.add_argument("--cell_w", type=int, default=320)
    p.add_argument("--cell_h", type=int, default=240)
    p.add_argument("--row_gap", type=int, default=6)
    p.add_argument("--col_gap", type=int, default=6)
    p.add_argument("--title", default="Digital twin: one frame in, the colony forward in time")
    p.add_argument("--min_per_frame", type=float, default=5.0)
    p.add_argument("--out", default="out/rule_time_grid.png")
    args = p.parse_args()

    panels_2d = []
    for label, sub, accent, kind, flags, suffix in ROWS:
        print("Row: %s" % label)
        if kind == "real":
            panels_2d.append(render_real_row(args))
        else:
            panels_2d.append(render_sim_row(args, flags, suffix))

    print("Tiling ...")
    build_grid(panels_2d, args.out, args.frames, args.title, args.min_per_frame,
               args.cell_w, args.cell_h, args.row_gap, args.col_gap)


if __name__ == "__main__":
    main()
