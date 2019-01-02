import sys
import os
import subprocess

class mpi_fork:
    def __init__(self, n):
        self._n = n

    def __enter__(self):
        try:
            # Import MPI4Py without initializing it
            import mpi4py.rc
            mpi4py.rc.initialize = False
            from mpi4py import MPI
            if MPI.Is_initialized():
                return "child"
        except:
            pass

        n = self._n
        if n <= 1:
            return "child"
        if os.getenv('D500_IN_MPI') is None:
            env = os.environ.copy()
            env.update(D500_IN_MPI="1")
            # n = 2
            args = ['mpiexec', '-np', str(n)]
            args += [sys.executable] + sys.argv
            print(' '.join(args))
            subprocess.check_call(args, env=env)
            return "parent"
        else:
            return "child"
        
    def __exit__(self, type, value, traceback):
        try:
            # Import MPI4Py without initializing it
            import mpi4py.rc
            mpi4py.rc.initialize = False
            from mpi4py import MPI

            mpi_init = MPI.Is_initialized()
        except:
            mpi_init = False
        if mpi_init or os.getenv('D500_IN_MPI') is not None:
                MPI.Finalize()

def mpi_end_barrier():
    """ Invokes a barrier and finalization if MPI is running, or nothing 
        otherwise. """
    # Import MPI4Py without initializing it
    import mpi4py.rc
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    if MPI.Is_initialized():
        MPI.COMM_WORLD.Barrier()
        MPI.Finalize()
