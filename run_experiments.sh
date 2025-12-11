#!/usr/bin/env bash
# run_experiments.sh
# Uso: ./run_experiments.sh
N=1000
DAYS=365
BETA=0.25
GAMMA=0.05
MU=0.001
INIT=100
SEED=42

OUTCSV="speedup_results.csv"
echo "np,wall_time" > $OUTCSV

for np in 1 2 4 8; do
    echo "Running np=$np ..."
    # clean previous times
    rm -f time_mpi_total.txt
    # run with mpirun, capture time
    mpirun -np $np python3 mpi_sir.py --n $N --days $DAYS --beta $BETA --gamma $GAMMA --mu $MU --init-infected $INIT --seed $SEED
    if [ -f time_mpi_total.txt ]; then
        t=$(cat time_mpi_total.txt)
    else
        # fallback: measure using /usr/bin/time
        t=$( ( /usr/bin/time -f "%e" mpirun -np $np python3 mpi_sir.py --n $N --days $DAYS --beta $BETA --gamma $GAMMA --mu $MU --init-infected $INIT --seed $SEED ) 2>&1 | tail -n1 )
    fi
    echo "${np},${t}" >> $OUTCSV
done

echo "Done. Results saved in $OUTCSV"
