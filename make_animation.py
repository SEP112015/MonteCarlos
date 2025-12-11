#!/usr/bin/env python3
"""
make_animation.py
Genera GIF/MP4 side-by-side comparando secuencial vs paralelo (rank 0 view).
Asume:
  frames_seq/seq_day_XXXX.png
  frames_mpi_rank0/mpi_day_XXXX.png
Salida:
  comparison_side_by_side.gif
"""
import imageio
from PIL import Image
import os
import glob
frames_seq = sorted(glob.glob("frames_seq/seq_day_*.png"))
frames_mpi = sorted(glob.glob("frames_mpi_rank0/mpi_day_*.png"))

n = min(len(frames_seq), len(frames_mpi))
outfile_gif = "comparison_side_by_side.gif"
frames = []
for i in range(n):
    a = Image.open(frames_seq[i])
    b = Image.open(frames_mpi[i])
    # resize to same height
    if a.size[1] != b.size[1]:
        # scale b to a height
        b = b.resize((int(b.size[0]*a.size[1]/b.size[1]), a.size[1]))
    # create combined image
    combined = Image.new('RGB', (a.size[0]+b.size[0], a.size[1]))
    combined.paste(a, (0,0))
    combined.paste(b, (a.size[0],0))
    frames.append(np.asarray(combined))

imageio.mimsave(outfile_gif, frames, fps=6)
print("Saved", outfile_gif)
