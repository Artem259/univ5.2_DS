# Parallel Matrix Multiplication using Cannon's Algorithm with MPI

## Description of the MPI Technology

**MPI (Message Passing Interface)** is a standard designed for organizing parallel computations based on message passing between processes. MPI provides a set of functions that allow the creation, synchronization, and communication between processes running in a distributed environment (e.g., on a cluster or a multiprocessor system).

MPI is implemented as libraries (e.g., `OpenMPI`, `MPICH`) and enables the scaling of computations by distributing tasks among processes.

In this project, the Python wrapper for MPI, **`mpi4py`**, is used.

---

## Description of MPI Functions Used in the Code

The following key MPI functions are used in the implementation of Cannon's algorithm:

### `MPI.COMM_WORLD`
> Creates a global communicator that includes all the processes participating in the program.
- **`Get_rank()`** — returns the unique identifier (rank) of the current process in the communicator.
- **`Get_size()`** — returns the total number of processes in the communicator.

### `comm.Create_cart(dims, periods, reorder)`
> Creates a Cartesian topology for processes (a grid) based on the provided dimensions.
- `dims` — dimensions of the grid (e.g., `[2, 2]` for 4 processes).
- `periods` — boolean values that determine the cyclicity of the grid along each coordinate.
- `reorder` — whether to allow MPI to reorder process ranks for optimization.

### `grid_comm.Get_coords(rank)`
> Returns the coordinates of a process in the Cartesian grid.

### `comm.Sub(remain_dims)`
> Creates a new communicator by extracting a subset of coordinates (e.g., a communicator for a row or column).
- `[False, True]` — fixes rows and changes columns (i.e., creates a row communicator).
- `[True, False]` — creates a column communicator.

### `comm.bcast(data, root)`
> Broadcasts the value `data` from the process `root` to all other processes in the communicator.

### `comm.Scatter(sendbuf, recvbuf, root)`
> Distributes parts of the array `sendbuf` from the root process to all processes.
- `recvbuf` — buffer where the received data will be written on each process.

### `comm.Gather(sendbuf, recvbuf, root)`
> Gathers data from all processes into the `recvbuf` array on the root process.

### `comm.Sendrecv_replace(buf, dest, sendtag, source, recvtag)`
> Simultaneously sends and receives a block of data:
- `buf` — buffer that will be both sent and replaced with received data.
- `dest`, `source` — ranks of the destination and source processes.
- `sendtag`, `recvtag` — message tags.

---

## Description of Cannon's Algorithm

**Cannon's Algorithm** is an efficient method for parallel matrix multiplication specifically designed for distributed computing in a grid topology of processes.

### Key Steps:

1. **Matrix Splitting**  
   Matrices A and B are split into blocks of size `block_size x block_size`. Each process receives one block of A and one block of B.

2. **Process Topology Initialization**  
   A Cartesian process grid of size √P x √P is created, where P is the number of processes. Separate communicators for rows and columns are also created.

3. **Initial Block Alignment**  
   The blocks are shifted left (matrix A) and up (matrix B) to ensure the correct initial placement for multiplication.

4. **Iterative Multiplication and Communication**  
   Over √P iterations:
   - Each process computes part of the result: `C_block += A_block * B_block`
   - A blocks are cyclically shifted left along rows, and B blocks are shifted up along columns.

5. **Result Collection**  
   All partial results of C blocks are gathered at the root process and form the final matrix.

### Advantages:

- Load balancing across processes
- Minimal data exchange in each iteration
- High efficiency with a large number of processes

---

## Structure of the `CannonMultiplication` Class

| Method | Purpose |
|--------|---------|
| `__init__` | Initializes multiplication parameters |
| `run()` | Main controlling method for execution |
| `_create_grid_communicators()` | Creates Cartesian topology and sub-communicators |
| `_process_initialization()` | Prepares matrices and buffers |
| `_matrix_scatter()` | Distributes matrices A and B across processes |
| `_data_distribution()` | Initializes block shifts |
| `_block_communication()` | Executes shifts using `Sendrecv_replace` |
| `_result_calculation()` | Computes the product of blocks |
| `_parallel_result_calculation()` | Full iteration of Cannon's algorithm |
| `_result_collection()` | Gathers partial results into the final matrix |

---

## Requirements for Running the Program

To run the program, you need:

- Python 3
- Installed libraries:
  - `mpi4py` (for parallel computations using MPI): `pip install mpi4py`
  - `numpy`: `pip install numpy`
- MPI parallel environment (e.g., `mpiexec` or `mpirun`)

### Launch Parameters

- `-n` — **number of processes**, which defines the process grid size √n × √n.  
  > The number of processes **must be a perfect square** (e.g., 4, 9, 16, ...).

- `-s` — **size of the square matrix N**, which defines the dimension `N x N`.  
  > The value of `-s` **must be divisible by the grid size**, i.e., `matrix_size % √n == 0`.

### Example

```bash
mpiexec -n 9 python main.py -s 18
```

- A 3×3 process grid is created.
- Each process works with a block of size 6×6 (because 18 / 3 = 6).

---