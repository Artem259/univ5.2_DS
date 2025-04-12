import numpy as np
import math
from mpi4py import MPI


class CannonMultiplication:
    def __init__(self, size, m1, m2, result):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.matrix_size = size
        self.m1 = m1
        self.m2 = m2
        self.result = result

        self.grid_coords = [0, 0]
        self.grid_size = None

    def run(self):
        self.grid_size = int(math.sqrt(self.size))
        if self.rank == 0:
            assert self.size == self.grid_size ** 2

        self._create_grid_communicators()
        self._process_initialization()
        self._data_distribution()
        self._parallel_result_calculation()
        self._result_collection()

    def _create_grid_communicators(self):
        dims = [self.grid_size, self.grid_size]
        periods = [False, False]
        self.grid_comm = self.comm.Create_cart(dims, periods=periods, reorder=True)
        self.grid_coords = self.grid_comm.Get_coords(self.rank)

        self.row_comm = self.grid_comm.Sub([False, True])
        self.col_comm = self.grid_comm.Sub([True, False])

    def _process_initialization(self):
        matrix_size = self.matrix_size if self.rank == 0 else None
        matrix_size = self.comm.bcast(matrix_size, root=0)

        self.block_size = matrix_size // self.grid_size

        self.a_block = np.zeros((self.block_size, self.block_size), dtype=np.float64)
        self.b_block = np.zeros((self.block_size, self.block_size), dtype=np.float64)
        self.c_block = np.zeros((self.block_size, self.block_size), dtype=np.float64)

        if self.rank == 0:
            self.a_matrix = self.m1.get_data_as_array()
            self.b_matrix = self.m2.get_data_as_array()
            self.c_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)

    def _matrix_scatter(self, full_matrix, block_matrix):
        row_buffer = np.zeros((self.block_size, self.matrix_size), dtype=np.float64)

        if self.grid_coords[1] == 0:
            self.col_comm.Scatter(full_matrix, row_buffer, root=0)

        for i in range(self.block_size):
            self.row_comm.Scatter(row_buffer[i], block_matrix[i], root=0)

    def _data_distribution(self):
        if self.rank == 0:
            self._matrix_scatter(self.a_matrix, self.a_block)
            self._matrix_scatter(self.b_matrix, self.b_block)
        else:
            self._matrix_scatter(None, self.a_block)
            self._matrix_scatter(None, self.b_block)

        self._block_communication(
            self.a_block, self.row_comm,
            self.grid_coords[1] - self.grid_coords[0],
            self.grid_coords[1] + self.grid_coords[0]
        )
        self._block_communication(
            self.b_block, self.col_comm,
            self.grid_coords[0] - self.grid_coords[1],
            self.grid_coords[0] + self.grid_coords[1]
        )

    def _block_communication(self, block, comm, dest, source):
        dest %= self.grid_size
        source %= self.grid_size
        comm.Sendrecv_replace(block, dest=dest, sendtag=0, source=source, recvtag=0)

    def _result_calculation(self):
        for i in range(self.block_size):
            for j in range(self.block_size):
                for k in range(self.block_size):
                    self.c_block[i, j] += self.a_block[i, k] * self.b_block[k, j]

    def _parallel_result_calculation(self):
        for _ in range(self.grid_size):
            self._result_calculation()
            self._block_communication(
                self.a_block, self.row_comm,
                self.grid_coords[1] - 1, self.grid_coords[1] + 1
            )
            self._block_communication(
                self.b_block, self.col_comm,
                self.grid_coords[0] - 1, self.grid_coords[0] + 1
            )

    def _result_collection(self):
        result_row = np.zeros((self.block_size, self.matrix_size), dtype=np.float64)
        for i in range(self.block_size):
            self.row_comm.Gather(self.c_block[i], result_row[i], root=0)

        if self.grid_coords[1] == 0:
            if self.rank == 0:
                self.col_comm.Gather(result_row, self.c_matrix, root=0)
            else:
                self.col_comm.Gather(result_row, None, root=0)

        if self.rank == 0:
            self.result.set_data_from_array(self.c_matrix)
