import ctypes
from typing import Set
import numpy as np

# Import MPI4Py without initializing it
try:
    import mpi4py.rc
    mpi4py.rc.initialize = False
    from mpi4py import MPI
except ImportError:
    print('WARNING: mpi4py not available, distributed optimization disabled')
    raise
    
class CommunicationNetwork(object):

    def __init__(self):
        if not MPI.Is_initialized():
            MPI.Init_thread()
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.size
        self.rank = self.comm.rank

    def get_comm_numpy_ptr(self):
        """
        Returns the global communicator as a numpy array bound to a pointer.
        """
        comm_ptr = MPI._addressof(self.comm)
        return np.array(comm_ptr)

    def sync_all_at_root(self, tensor, gather_func=np.sum):
        """
        Send tensor to root node (usually a parameter server) and 
        synchronize gathered tensor back at the nodes.
        """
        # Gather tensor at PS
        received_tensor = np.empty([self.size - 1] + list(tensor.shape), dtype=np.float32)
        self.comm.Gatherv(tensor, [received_tensor, [0] + (self.size - 1) * [tensor.size]], root=0)

        # Tensor to sync back to ranks
        sync_back_tensor = np.empty(tensor.shape)
        if self.rank == 0:
            sync_back_tensor = gather_func(received_tensor, axis=0)

        self.comm.bcast(sync_back_tensor, root=0)

        return sync_back_tensor

    def gather_at_root(self, tensor):
        """
        Gather tensors at root rank.
        @param tensor Tensor to gather
        @return Gathered tensor
        """
        received_tensor = np.empty([self.size - 1] + list(tensor.shape), dtype=np.float32)
        self.comm.Gatherv(tensor, [received_tensor, [0] + (self.size - 1) * [tensor.size]], root=0)
        return received_tensor

    def sync_all_with_root(self, tensor):
        self.comm.Bcast(tensor, root=0)

    def send_to_root(self, tensor, tag):
        """ Blocking send of a tensor to the root node.
            @param tensor The data to send.
            @param tag Additional information to include with the message.
        """
        if self.rank != 0:
            self.comm.send(tensor, dest=0, tag=tag)

    def wait_for_any_rank(self):
        """
        Waits for any message from any rank.
        @return A 3-tuple of (tensor, source rank, tag)
        """
        status = MPI.Status()
        tensor = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        return tensor, status.source, status.tag

    def send_to_rank(self, tensor, dest, tag):
        """
        Blocking send to specific rank
        @param tensor tensor to send
        @param dest specifies rank
        @param tag Message information
        """
        if self.rank == 0:
            self.comm.send(tensor, dest=dest, tag=tag)

    def sync_all(self, tensor, reduce_func=MPI.SUM):
        """
        All ranks communicate via allreduce and returns the summarized tensor.
        @param tensor The data to reduce.
        @param reduce_func MPI function with which to reduce tensor.
        @return Reduced tensor.
        """
        received_tensor = tensor.copy()
        self.comm.Allreduce(tensor, received_tensor, op=reduce_func)

        return received_tensor

    def wait_for_root(self):
        """
        Wait for message from root. Normally these are the updated parameters.
        @return The sent data from the root.
        """
        tensor = self.comm.recv(source=0, tag=MPI.ANY_TAG)
        return tensor

    def reduce_from_neighbors(self, tensor, reduce_func=sum):
        """
        Reduces information between neighbors.
        @param tensor The data to reduce.
        @param reduce_func MPI function with which to reduce tensor.
        @return Reduced tensor.
        """
        left = (self.rank - 1) % self.size
        right = (self.rank + 1) % self.size
        left_tensor = self.comm.sendrecv(tensor, dest=right, source=left)
        right_tensor = self.comm.sendrecv(tensor, dest=left, source=right)
        return reduce_func((tensor, left_tensor, right_tensor))
