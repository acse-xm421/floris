from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generate data on each process
data = rank + 1

# Perform the reduction (sum) across all processes
total_sum = comm.reduce(data, op=MPI.SUM, root=0)

# Print results
if rank == 0:
    print("Total sum:", total_sum)

# Finalize MPI
MPI.Finalize()
