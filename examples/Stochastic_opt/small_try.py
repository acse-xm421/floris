from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Print rank and size
print(f"Hello from rank {rank} out of {size} processes!")

# Synchronize processes
comm.Barrier()

# Finalize MPI
MPI.Finalize()
