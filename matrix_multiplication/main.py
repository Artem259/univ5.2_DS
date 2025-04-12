import argparse
import sys
import numpy as np
from mpi4py import MPI

from matrix import Matrix
from CannonMultiplication import CannonMultiplication


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--matrix_size', type=int, default=None,
        help="Size of the NxN matrix (must be divisible by grid size)"
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    grid_size = int(np.sqrt(size))
    if grid_size ** 2 != size:
        if rank == 0:
            print(f"Error: Number of processes ({size}) is not a perfect square.")
        sys.exit()

    matrix_size = args.matrix_size or (grid_size * 3)

    if matrix_size % grid_size != 0:
        if rank == 0:
            print(
                f"Error: Matrix size {matrix_size} must be divisible by grid size {grid_size}."
            )
        sys.exit()

    expected_result = None
    if rank == 0:
        matrix_a = Matrix(size=matrix_size)
        matrix_b = Matrix(size=matrix_size)
        matrix_c = Matrix(data=np.zeros((matrix_size, matrix_size), dtype=np.float64))
        expected_result = matrix_a.get_data_as_array() @ matrix_b.get_data_as_array()
    else:
        matrix_a = Matrix()
        matrix_b = Matrix()
        matrix_c = Matrix()

    cannon = CannonMultiplication(
        size=matrix_size, m1=matrix_a, m2=matrix_b, result=matrix_c
    )
    cannon.run()

    if rank == 0:
        block_size = matrix_size // grid_size
        print(f"Matrix size: {matrix_size}x{matrix_size}")
        print(f"Processor grid: {grid_size}x{grid_size}")
        print(f"Block size: {block_size}x{block_size}")

        print()
        if np.allclose(matrix_c.get_data_as_array(), expected_result):
            print("Test with NumPy: Results match.")
        else:
            print("Test with NumPy: Results do not match.")


if __name__ == "__main__":
    main()
