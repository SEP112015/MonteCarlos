#!/usr/bin/env python3
"""
seq_sir.py
Simulación secuencial SIR+Death en grilla 2D.
Uso:
    python3 seq_sir.py --n 1000 --days 365 --beta 0.25 --gamma 0.05 --mu 0.001
Opciones:
    --init-infected: número inicial de infectados aleatorios (por defecto 10)
    --seed: semilla aleatoria para reproducibilidad
Salida:
    - frames/seq_day_%04d.png  (opcional: para animación)
    - stats_seq.csv  (tiempos y conteos por día)
"""
import numpy as np
import argparse
import time
import csv
import os
from scipy.signal import convolve2d
import imageio

STATE_S = 0
STATE_I = 1
STATE_R = 2
STATE_D = 3

KERNEL = np.array([[1,1,1],
                   [1,0,1],
                   [1,1,1]], dtype=np.uint8)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="tamaño de la grilla (n x n)")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--beta", type=float, required=True, help="prob. transmisión por vecino por día")
    p.add_argument("--gamma", type=float, required=True, help="prob. recuperación por día")
    p.add_argument("--mu", type=float, required=True, help="prob. muerte por día")
    p.add_argument("--init-infected", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-frames", action="store_true")
    return p.parse_args()

def step(grid, beta, gamma, mu):
    # grid: uint8 array of states
    infected_mask = (grid == STATE_I).astype(np.uint8)
    # conteo de vecinos infectados (Moore)
    neigh_inf = convolve2d(infected_mask, KERNEL, mode='same', boundary='fill', fillvalue=0)
    susceptible_mask = (grid == STATE_S)
    # prob de infectarse por celda (S): 1 - (1-beta)**k
    k = neigh_inf[susceptible_mask]
    if k.size > 0:
        p_inf = 1.0 - np.power((1.0 - beta), k)
        rand_vals = np.random.rand(p_inf.size)
        new_infected_indices = np.where(rand_vals < p_inf)[0]
        # Aplica S -> I según new_infected_indices
        sus_positions = np.flatnonzero(susceptible_mask)
        infections = sus_positions[new_infected_indices]
    else:
        infections = np.array([], dtype=np.int64)

    # Recuperaciones y muertes de I actuales
    infect_positions = np.flatnonzero(infected_mask)
    r_rand = np.random.rand(infect_positions.size)
    recoveries = infect_positions[r_rand < gamma]
    deaths = infect_positions[(r_rand >= gamma) & (r_rand < gamma + mu)]

    # Crear nuevo grid
    new_grid = grid.copy()
    # Aplicar S->I
    if infections.size > 0:
        new_grid.flat[infections] = STATE_I
    # Aplicar I->R
    if recoveries.size > 0:
        new_grid.flat[recoveries] = STATE_R
    # Aplicar I->D
    if deaths.size > 0:
        new_grid.flat[deaths] = STATE_D

    new_infections_count = infections.size
    return new_grid, new_infections_count

def main():
    args = parse_args()
    np.random.seed(args.seed)
    n = args.n
    grid = np.zeros((n,n), dtype=np.uint8)  # all susceptible
    # initial infections random
    idx = np.random.choice(n*n, size=args.init_infected, replace=False)
    grid.flat[idx] = STATE_I

    stats = []
    infected_accum = args.init_infected

    out_dir = "frames_seq"
    if args.save_frames:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for day in range(args.days):
        current_counts = [np.count_nonzero(grid==s) for s in (STATE_S, STATE_I, STATE_R, STATE_D)]
        # step
        grid, new_inf = step(grid, args.beta, args.gamma, args.mu)
        infected_accum += new_inf
        # recompute counts after update
        s,i,r,d = [np.count_nonzero(grid==s0) for s0 in (STATE_S, STATE_I, STATE_R, STATE_D)]
        # estimate R_t safely
        R_t = new_inf / i if i>0 else 0.0
        stats.append((day, s, i, r, d, new_inf, infected_accum, R_t))
        if args.save_frames:
            # simple color mapping: S=0 white, I=red, R=green, D=black
            import matplotlib.pyplot as plt
            cmap = {STATE_S:255, STATE_I:200, STATE_R:100, STATE_D:0}
            arr = np.vectorize(cmap.get)(grid).astype(np.uint8)
            plt.imsave(f"{out_dir}/seq_day_{day:04d}.png", arr, cmap="gray", vmin=0, vmax=255)
        # small progress
        if day % 50 == 0:
            print(f"[seq] day {day:4d}  S={s} I={i} R={r} D={d} new_inf={new_inf}")
    t1 = time.time()
    total_time = t1 - t0
    # save stats
    with open("stats_seq.csv","w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["day","S","I","R","D","new_inf","infected_accum","R_t"])
        w.writerows(stats)
    print("Finished seq run. time(s) =", total_time)
    with open("time_seq.txt","w") as f:
        f.write(str(total_time))

if __name__ == "__main__":
    main()
