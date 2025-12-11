#!/usr/bin/env python3
"""
plot_speedup.py
Lee speedup_results.csv y genera walltime_vs_cores.png y speedup.png
Uso:
    python plot_speedup.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

fn = "speedup_results.csv"
if not os.path.exists(fn):
    print("No se encontró", fn)
    sys.exit(1)

df = pd.read_csv(fn)
# Convertir wall_time to numeric (handle NA)
df['wall_time'] = pd.to_numeric(df['wall_time'], errors='coerce')

# Drop rows with NA wall_time
df = df.dropna(subset=['wall_time']).sort_values('np')

if df.empty:
    print("No hay datos válidos en", fn)
    sys.exit(1)

# baseline (np==1)
if 1 in df['np'].values:
    t1 = float(df.loc[df['np'] == 1, 'wall_time'].values[0])
else:
    t1 = float(df['wall_time'].iloc[0])

df['speedup'] = t1 / df['wall_time']

plt.figure(figsize=(6,4))
plt.plot(df['np'], df['wall_time'], marker='o')
plt.xlabel("num cores (np)")
plt.ylabel("wall time (s)")
plt.title("Wall time vs cores")
plt.grid(True)
plt.tight_layout()
plt.savefig("walltime_vs_cores.png", dpi=200)
print("Saved walltime_vs_cores.png")

plt.figure(figsize=(6,4))
plt.plot(df['np'], df['speedup'], marker='o')
plt.xlabel("num cores (np)")
plt.ylabel("speed-up")
plt.title("Speed-up (strong scaling)")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup.png", dpi=200)
print("Saved speedup.png")
