"""
side_by_side_combiner.py
------------------------
Stitch real.gif (real omnipose masks) and sim.gif (Viva-munk simulation masks)
into a single side-by-side GIF with a frame/time header and an IPTG marker
that lights up at and after the media-switch frame.

    python side_by_side_combiner.py path/to/real.gif path/to/sim.gif

Optional flags:
    --output         path to output GIF (default out/real_vs_sim.gif)
    --panel_size     square size in pixels for each panel (default 600)
    --frame_max      last frame to include, inclusive (default 130)
    --iptg_frame     frame at which IPTG arrives (default 28)
    --frame_interval seconds per frame (default 300)
    --fps            output frames per second (default 10)
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont


HEADER_PX = 56
PANEL_GAP_PX = 8
LABEL_PX = 28


def load_gif_frames(path):
    im = Image.open(path)
    frames = []
    n = 0
    while True:
        try:
            im.seek(n)
            frames.append(im.convert('L').copy())
            n += 1
        except EOFError:
            break
    return frames


def letterbox(frame, target_size):
    src = frame
    sw, sh = src.size
    scale = min(target_size / sw, target_size / sh)
    new_w = max(1, int(round(sw * scale)))
    new_h = max(1, int(round(sh * scale)))
    resized = src.resize((new_w, new_h), Image.NEAREST)
    canvas = Image.new('L', (target_size, target_size), 0)
    canvas.paste(resized, ((target_size - new_w) // 2,
                           (target_size - new_h) // 2))
    return canvas


def font(size):
    for cand in ['/System/Library/Fonts/Menlo.ttc',
                 '/System/Library/Fonts/Helvetica.ttc',
                 '/System/Library/Fonts/Supplemental/Courier New.ttf']:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_frame(real_panel, sim_panel, frame_idx, iptg_frame,
                  frame_interval, panel_size,
                  label_left='real', label_right='sim'):
    width = panel_size * 2 + PANEL_GAP_PX
    height = HEADER_PX + panel_size + LABEL_PX
    img = Image.new('RGB', (width, height), (10, 10, 10))
    draw = ImageDraw.Draw(img)

    panel_y = HEADER_PX
    img.paste(real_panel.convert('RGB'), (0, panel_y))
    img.paste(sim_panel.convert('RGB'), (panel_size + PANEL_GAP_PX, panel_y))

    iptg_on = frame_idx >= iptg_frame
    t_min = (frame_idx * frame_interval) / 60.0
    header = f"frame {frame_idx:>3} / {frame_interval/60:.0f} min/step  |  t = {t_min:6.1f} min"
    draw.text((12, 12), header, fill=(230, 230, 230), font=font(18))
    if iptg_on:
        badge = "+IPTG"
        f = font(20)
        bbox = draw.textbbox((0, 0), badge, font=f)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bx = width - bw - 18
        by = 14
        draw.rectangle([bx - 8, by - 4, bx + bw + 8, by + bh + 4],
                       fill=(0, 110, 50))
        draw.text((bx, by), badge, fill=(255, 255, 255), font=f)

    label_y = HEADER_PX + panel_size + 4
    f_label = font(18)
    bbox_l = draw.textbbox((0, 0), label_left, font=f_label)
    bbox_r = draw.textbbox((0, 0), label_right, font=f_label)
    wl = bbox_l[2] - bbox_l[0]
    wr = bbox_r[2] - bbox_r[0]
    draw.text((panel_size // 2 - wl // 2, label_y), label_left,
              fill=(220, 220, 220), font=f_label)
    draw.text((panel_size + PANEL_GAP_PX + panel_size // 2 - wr // 2, label_y),
              label_right, fill=(220, 220, 220), font=f_label)

    return img


def main():
    parser = argparse.ArgumentParser(
        description='Side-by-side combiner for real.gif and sim.gif')
    parser.add_argument('real', help='Path to real.gif')
    parser.add_argument('sim', help='Path to sim.gif')
    parser.add_argument('--output', default='out/real_vs_sim.gif')
    parser.add_argument('--panel_size', type=int, default=600)
    parser.add_argument('--frame_max', type=int, default=130)
    parser.add_argument('--iptg_frame', type=int, default=28)
    parser.add_argument('--frame_interval', type=float, default=300)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--label_left', default='real')
    parser.add_argument('--label_right', default='sim')
    args = parser.parse_args()

    print(f"Loading {args.real} ...")
    real = load_gif_frames(args.real)
    print(f"  {len(real)} frames, size {real[0].size}")
    print(f"Loading {args.sim} ...")
    sim = load_gif_frames(args.sim)
    print(f"  {len(sim)} frames, size {sim[0].size}")

    n = min(len(real), len(sim), args.frame_max + 1)
    print(f"Composing {n} side-by-side frames ...")

    frames = []
    for i in range(n):
        real_p = letterbox(real[i], args.panel_size)
        sim_p = letterbox(sim[i], args.panel_size)
        frames.append(compose_frame(
            real_p, sim_p, i, args.iptg_frame,
            args.frame_interval, args.panel_size,
            label_left=args.label_left, label_right=args.label_right,
        ))
        if i % 10 == 0:
            print(f"  composed frame {i}/{n-1}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    duration_ms = int(1000 / args.fps)
    pal = [f.convert('P', palette=Image.ADAPTIVE) for f in frames]
    pal[0].save(
        args.output,
        save_all=True,
        append_images=pal[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"\nDone! GIF: {args.output}  ({len(pal)} frames @ {args.fps} fps)")


if __name__ == '__main__':
    main()
