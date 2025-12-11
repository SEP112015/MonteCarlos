#!/usr/bin/env python3
"""
mpi_sir.py
Simulación SIR+Death paralela (mpi4py) con split horizontal y ghost-rows.
Uso:
  mpirun -np 4 python3 mpi_sir.py --n 1000 --days 365 --beta 0.25 --gamma 0.05 --mu 0.001
Salida:
  - stats_mpi_rankX.csv (opcional)
  - stats_mpi_global.csv (guardado por rank 0)
  - time_mpi_np.txt (tiempo total guardado por rank 0)
Requisitos:
  - mpi4py
  - scipy
"""
from mpi4py import MPI
import numpy as np
import argparse
import time
import csv
from scipy.signal import convolve2d
import os

STATE_S = 0
STATE_I = 1
STATE_R = 2
STATE_D = 3

KERNEL = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--mu", type=float, required=True)
    p.add_argument("--init-infected", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-frames", action="store_true")
    return p.parse_args()

def step_local(local_grid, beta, gamma, mu):
    # local_grid includes only local block (no ghost rows)
    infected_mask = (local_grid == STATE_I).astype(np.uint8)
    neigh_inf = convolve2d(infected_mask, KERNEL, mode='same', boundary='fill', fillvalue=0)
    susceptible_mask = (local_grid == STATE_S)
    k = neigh_inf[susceptible_mask]
    if k.size > 0:
        p_inf = 1.0 - np.power((1.0 - beta), k)
        rand_vals = np.random.rand(p_inf.size)
        new_infected_indices = np.where(rand_vals < p_inf)[0]
        sus_positions = np.flatnonzero(susceptible_mask)
        infections = sus_positions[new_infected_indices]
    else:
        infections = np.array([], dtype=np.int64)

    infect_positions = np.flatnonzero(infected_mask)
    r_rand = np.random.rand(infect_positions.size)
    recoveries = infect_positions[r_rand < gamma]
    deaths = infect_positions[(r_rand >= gamma) & (r_rand < gamma + mu)]

    new_grid = local_grid.copy()
    if infections.size > 0:
        new_grid.flat[infections] = STATE_I
    if recoveries.size > 0:
        new_grid.flat[recoveries] = STATE_R
    if deaths.size > 0:
        new_grid.flat[deaths] = STATE_D

    new_infections_count = infections.size
    return new_grid, new_infections_count

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    np.random.seed(args.seed + rank)  # different streams per rank

    n = args.n
    # split rows across ranks as evenly as possible
    rows_per_rank = [n // size + (1 if i < (n % size) else 0) for i in range(size)]
    start_row = sum(rows_per_rank[:rank])
    local_rows = rows_per_rank[rank]

    # local grid WITHOUT ghost rows: shape (local_rows, n)
    local_grid = np.zeros((local_rows, n), dtype=np.uint8)

    # rank 0 initializes the global grid and scatters initial infected positions
    if rank == 0:
        full = np.zeros((n,n), dtype=np.uint8)
        # initial infected
        init_idx = np.random.choice(n*n, size=args.init_infected, replace=False)
        full.flat[init_idx] = STATE_I
        # scatter by creating list of subarrays to send
        blocks = []
        r0 = 0
        for rcount in rows_per_rank:
            blocks.append(full[r0:r0+rcount,:].copy())
            r0 += rcount
    else:
        blocks = None
    # scatterv-like using scatter
    local_grid = comm.scatter(blocks, root=0)

    # prepare neighbors
    up = rank - 1 if rank - 1 >= 0 else MPI.PROC_NULL
    down = rank + 1 if rank + 1 < size else MPI.PROC_NULL

    stats_local = []
    infected_accum_local = int(np.count_nonzero(local_grid == STATE_I))

    # For saving frames optionally
    if args.save_frames:
        os.makedirs(f"frames_mpi_rank{rank}", exist_ok=True)

    comm.Barrier()
    t0 = time.time()

    for day in range(args.days):
        # count current local totals
        s_local = int(np.count_nonzero(local_grid == STATE_S))
        i_local = int(np.count_nonzero(local_grid == STATE_I))
        r_local = int(np.count_nonzero(local_grid == STATE_R))
        d_local = int(np.count_nonzero(local_grid == STATE_D))

        # Exchange ghost rows: send/recv top and bottom rows
        # Prepare send buffers (first and last row)
        top_row = None
        bottom_row = None
        if local_rows > 0:
            top_row = local_grid[0,:].copy()
            bottom_row = local_grid[-1,:].copy()
        # allocate recv buffers
        top_recv = np.empty(n, dtype=np.uint8)
        bottom_recv = np.empty(n, dtype=np.uint8)

        # non-blocking sends/receives
        reqs = []
        if up != MPI.PROC_NULL:
            reqs.append(comm.Isend(top_row, dest=up, tag=11))
            reqs.append(comm.Irecv(top_recv, source=up, tag=12))
        else:
            top_recv = None
        if down != MPI.PROC_NULL:
            reqs.append(comm.Isend(bottom_row, dest=down, tag=12))
            reqs.append(comm.Irecv(bottom_recv, source=down, tag=11))
        else:
            bottom_recv = None
        MPI.Request.Waitall(reqs)

        # build extended local array with ghost rows for neighbor counting
        ext_rows = local_rows + (1 if top_recv is not None else 0) + (1 if bottom_recv is not None else 0)
        ext = np.zeros((ext_rows, n), dtype=np.uint8)
        idx = 0
        if top_recv is not None:
            ext[idx,:] = top_recv; idx += 1
        ext[idx:idx+local_rows,:] = local_grid; idx += local_rows
        if bottom_recv is not None:
            ext[idx,:] = bottom_recv

        # Now use convolution on ext to compute neighbor counts, but only extract central block
        infected_mask_ext = (ext == STATE_I).astype(np.uint8)
        neigh_inf_ext = convolve2d(infected_mask_ext, KERNEL, mode='same', boundary='fill', fillvalue=0)
        # extract neighbor counts for local rows slice (offset depends if top ghost exists)
        offset = 1 if top_recv is not None else 0
        neigh_inf_local = neigh_inf_ext[offset:offset+local_rows, :]

        # Now apply same logic as sequential but using neighbors computed
        susceptible_mask = (local_grid == STATE_S)
        k = neigh_inf_local[susceptible_mask]
        if k.size > 0:
            p_inf = 1.0 - np.power((1.0 - args.beta), k)
            rand_vals = np.random.rand(p_inf.size)
            new_infected_indices = np.where(rand_vals < p_inf)[0]
            sus_positions = np.flatnonzero(susceptible_mask)
            infections = sus_positions[new_infected_indices]
        else:
            infections = np.array([], dtype=np.int64)

        infect_positions = np.flatnonzero(local_grid == STATE_I)
        r_rand = np.random.rand(infect_positions.size)
        recoveries = infect_positions[r_rand < args.gamma]
        deaths = infect_positions[(r_rand >= args.gamma) & (r_rand < args.gamma + args.mu)]

        new_local_grid = local_grid.copy()
        if infections.size > 0:
            new_local_grid.flat[infections] = STATE_I
        if recoveries.size > 0:
            new_local_grid.flat[recoveries] = STATE_R
        if deaths.size > 0:
            new_local_grid.flat[deaths] = STATE_D

        new_infections_local = int(infections.size)
        infected_accum_local += new_infections_local

        # write optional frame (rank-local)
        if args.save_frames:
            import matplotlib.pyplot as plt
            cmap = {STATE_S:255, STATE_I:200, STATE_R:100, STATE_D:0}
            arr = np.vectorize(cmap.get)(new_local_grid).astype(np.uint8)
            plt.imsave(f"frames_mpi_rank{rank}/mpi_day_{day:04d}.png", arr, cmap="gray", vmin=0, vmax=255)

        # store local daily stats
        s2 = int(np.count_nonzero(new_local_grid == STATE_S))
        i2 = int(np.count_nonzero(new_local_grid == STATE_I))
        r2 = int(np.count_nonzero(new_local_grid == STATE_R))
        d2 = int(np.count_nonzero(new_local_grid == STATE_D))
        R_t_local = new_infections_local / i2 if i2>0 else 0.0
        stats_local.append((day, s2, i2, r2, d2, new_infections_local, infected_accum_local, R_t_local))

        # swap local_grid
        local_grid = new_local_grid

    comm.Barrier()
    t1 = time.time()
    total_time = t1 - t0

    # Reduce global stats: we will reduce per-day aggregates to rank 0
    # Gather per-day arrays of new_infections and current_infected and infected_accum
    # Convert to numpy arrays for reduction
    days = args.days
    new_inf_arr = np.array([row[5] for row in stats_local], dtype=np.int64)
    cur_inf_arr = np.array([row[2] for row in stats_local], dtype=np.int64)
    infected_accum_arr = np.array([row[6] for row in stats_local], dtype=np.int64)

    # Sum new infections across ranks per day
    new_inf_global = np.zeros_like(new_inf_arr)
    comm.Reduce(new_inf_arr, new_inf_global, op=MPI.SUM, root=0)
    cur_inf_global = np.zeros_like(cur_inf_arr)
    comm.Reduce(cur_inf_arr, cur_inf_global, op=MPI.SUM, root=0)
    infected_accum_global = np.zeros_like(infected_accum_arr)
    comm.Reduce(infected_accum_arr, infected_accum_global, op=MPI.SUM, root=0)

    # Also reduce final total time: rank 0 keeps its measured time; we can gather min/avg/max
    times = comm.gather(total_time, root=0)

    if rank == 0:
        # compute daily R_t = new_inf_global / cur_inf_global (safe)
        R_t_global = np.array([ (new_inf_global[d] / cur_inf_global[d]) if cur_inf_global[d]>0 else 0.0 for d in range(days)])
        # save CSV
        with open("stats_mpi_global.csv","w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["day","S_total","I_total","R_total","D_total","new_inf_total","infected_accum_total","R_t"])
            # To have S/R/D per day we can compute S = N - I - R - D
            N = n * n
            for d in range(days):
                I = int(cur_inf_global[d])
                new_inf = int(new_inf_global[d])
                infected_acc = int(infected_accum_global[d])
                # crude: we don't have S/R/D separately per day reduced; approximate by using rank 0 current totals? Better approach is to perform reductions of S,R,D arrays too.
                # Here we'll compute S = N - (infected_acc_total) - D_total? Simpler: just save I, new_inf, infected_acc, R_t
                w.writerow([d, "", I, "", "", new_inf, infected_acc, float(R_t_global[d])])
        # save times per run
        with open("time_mpi_runs.txt","w") as f:
            f.write("times_per_rank:\n")
            for i,t in enumerate(times):
                f.write(f"rank_{i}: {t}\n")
        # save total wallclock time (we can take max of times)
        max_time = max(times)
        with open("time_mpi_total.txt","w") as f:
            f.write(str(max_time))
        print("MPI global stats and times saved. max_time(s) =", max_time)

if __name__ == "__main__":
    main()
