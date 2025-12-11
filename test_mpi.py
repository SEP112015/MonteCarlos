from mpi4py import MPI
print("Hello from rank", MPI.COMM_WORLD.Get_rank())
