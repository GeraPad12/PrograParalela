from mpi4py import MPI
import numpy as np

def quickselect(arr, k):
    if len(arr) == 1:
        return arr[0]
    pivot = arr[np.random.randint(len(arr))]
    lows = [el for el in arr if el < pivot]
    highs = [el for el in arr if el > pivot]
    pivots = [el for el in arr if el == pivot]
    
    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots))

def sample_select(arr, k, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Scatter the data to all processes
    local_data = np.array_split(arr, size)[rank]
    
    # Each process performs quickselect on its local data
    local_result = quickselect(local_data.tolist(), min(k, len(local_data)-1))

    # Gather all local results at the root process
    all_results = comm.gather(local_result, root=0)

    # Root process performs quickselect on the gathered results
    if rank == 0:
        return quickselect(all_results, k)
    else:
        return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Example data and k
    data = np.random.randint(0, 100, 100)
    k = 10

    # Broadcast the data and k to all processes
    data = comm.bcast(data if rank == 0 else None, root=0)
    k = comm.bcast(k if rank == 0 else None, root=0)

    # Perform sample select
    result = sample_select(data, k, comm)

    # Print the result in the root process
    if rank == 0:
        print(f"The {k+1}-th smallest element is: {result}")

if __name__ == "__main__":
    main()