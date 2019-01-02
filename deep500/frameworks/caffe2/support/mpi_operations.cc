#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#include "mpi_operations.h"
#include <mpi.h>

namespace caffe2 {

    template <typename T, class Context>
    class DMpiAllreduceOp final : public Operator<CPUContext> {
    public:
        USE_OPERATOR_FUNCTIONS(CPUContext);

        DMpiAllreduceOp(const OperatorDef& operator_def, Workspace* ws) : Operator<CPUContext>(operator_def, ws),
                                                                     ws_(ws) {
        }


        bool RunOnDevice() override  {
            const TensorCPU& comm_ptr = Input(1);
            long ptr = *comm_ptr.template data<long>();

            int size, rank;
            MPI_Comm  comm = *((MPI_Comm *) ptr);
            MPI_Comm_size(comm, &size);
            MPI_Comm_rank(comm, &rank);


            auto& X = Input(0);
            auto* Y = Output(0);
            void* source;

            Y->ResizeLike(X);
            if (Y->template mutable_data<T>() == X.template data<T>()) {
                source = MPI_IN_PLACE;
            } else {
                source = const_cast<T*>(X.template data<T>());
            }

            MPI_Allreduce(source, Y->template mutable_data<T>(), X.size(),MPIDataTypeWrapper<T>::type(), MPI_SUM, comm);

            return true;
        }


    protected:
        int arg1_;
        float arg2_;
        Workspace* ws_;

    };


    template <typename T, class Context>
    class DMpiAllreduceMeanOp final : public Operator<CPUContext> {
    public:
        USE_OPERATOR_FUNCTIONS(CPUContext);

        DMpiAllreduceMeanOp(const OperatorDef& operator_def, Workspace* ws) : Operator<CPUContext>(operator_def, ws),
                                                                          ws_(ws) {
        }


        bool RunOnDevice() override  {
            const TensorCPU& comm_ptr = Input(1);
            long ptr = *comm_ptr.template data<long>();

            int world_size, rank;
            MPI_Comm  comm = *((MPI_Comm *) ptr);
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &rank);


            auto& X = Input(0);
            auto* Y = Output(0);
            void* source;

            Y->ResizeLike(X);
            if (Y->template mutable_data<T>() == X.template data<T>()) {
                source = MPI_IN_PLACE;
            } else {
                source = const_cast<T*>(X.template data<T>());
            }

            MPI_Allreduce(source, Y->template mutable_data<T>(), X.size(), MPIDataTypeWrapper<T>::type(), MPI_SUM, comm);

            T* Y_Mut = Y->template mutable_data<T>();

            math::Scale(
                    Y->size(),
                    1.0f / world_size,
                    Y_Mut,
                    Y_Mut,
                    &context_);

            return true;
        }


    protected:
        int arg1_;
        float arg2_;
        Workspace* ws_;

    };

    template <typename T, class Context>
    class DMpiReduceMeanOp final : public Operator<CPUContext> {
    public:
        USE_OPERATOR_FUNCTIONS(CPUContext);

        DMpiReduceMeanOp(const OperatorDef& operator_def, Workspace* ws) : Operator<CPUContext>(operator_def, ws),
                                                                              ws_(ws) {
        }


        bool RunOnDevice() override  {
            const TensorCPU& comm_ptr = Input(1);
            long ptr = *comm_ptr.template data<long>();

            int world_size, rank;
            MPI_Comm  comm = *((MPI_Comm *) ptr);
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &rank);


            auto& X = Input(0);
            auto* Y = Output(0);
            void* source;

            Y->ResizeLike(X);
            if (Y->template mutable_data<T>() == X.template data<T>()) {
                source = MPI_IN_PLACE;
            } else {
                source = const_cast<T*>(X.template data<T>());
            }

            MPI_Reduce(source, Y->template mutable_data<T>(), X.size(), MPIDataTypeWrapper<T>::type(), MPI_SUM,0, comm);

            if (rank == 0) {
                T* Y_Mut = Y->template mutable_data<T>();

                math::Scale(
                        Y->size(),
                        1.0f / world_size,
                        Y_Mut,
                        Y_Mut,
                        &context_);
            }

            return true;
        }


    protected:
        int arg1_;
        float arg2_;
        Workspace* ws_;

    };


    template <typename T, class Context>
    class DMpiGatherOp final : public Operator<CPUContext> {
    public:
        USE_OPERATOR_FUNCTIONS(CPUContext);

        DMpiGatherOp(const OperatorDef& operator_def, Workspace* ws) : Operator<CPUContext>(operator_def, ws),
                                                                              ws_(ws) {
        }


        bool RunOnDevice() override  {
            const TensorCPU& comm_ptr = Input(1);
            long ptr = *comm_ptr.template data<long>();

            int world_size, rank;
            MPI_Comm  comm = *((MPI_Comm *) ptr);
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &rank);


            auto& X = Input(0);
            auto* Y = Output(0);
            void* source;

            auto Y_dims = X.dims();
            Y_dims.insert(Y_dims.begin(), world_size);
            Y->Resize(Y_dims);

            source = const_cast<T*>(X.template data<T>());

            MPI_Gather(source, X.size(), MPIDataTypeWrapper<T>::type(),
                       Y->template mutable_data<T>(), X.size(), MPIDataTypeWrapper<T>::type(),
                               0, comm);

            return true;
        }


    protected:
        int arg1_;
        float arg2_;
        Workspace* ws_;

    };

    template <typename T, class Context>
    class DMpiBroadcastOp final : public Operator<CPUContext> {
    public:
        USE_OPERATOR_FUNCTIONS(CPUContext);

        DMpiBroadcastOp(const OperatorDef& operator_def, Workspace* ws) : Operator<CPUContext>(operator_def, ws),
                                                                       ws_(ws) {
        }


        bool RunOnDevice() override  {
            const TensorCPU& comm_ptr = Input(1);
            long ptr = *comm_ptr.template data<long>();

            int world_size, rank;
            MPI_Comm  comm = *((MPI_Comm *) ptr);
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &rank);


            auto& X = Input(0);
            auto* Y = Output(0);

            CAFFE_ENFORCE(
                    Y->size() > 0,
                    "Broadcast op uses in-place operation so the output "
                    "should be already allocated.");

            MPI_Bcast(Y->raw_mutable_data(), Y->nbytes(), MPIDataTypeWrapper<char>::type(), 0, comm);

            return true;
        }


    protected:
        int arg1_;
        float arg2_;
        Workspace* ws_;

    };

    template <typename T, class Context>
    class DMpiRecvOp final : public Operator<CPUContext> {
    public:
        USE_OPERATOR_FUNCTIONS(CPUContext);

        DMpiRecvOp(const OperatorDef& operator_def, Workspace* ws) : Operator<CPUContext>(operator_def, ws),
                OP_SINGLE_ARG(int, "source", source_, MPI_ANY_SOURCE),
                OP_SINGLE_ARG(int, "tag", tag_, MPI_ANY_TAG),
                ws_(ws) {
        }


        bool RunOnDevice() override  {
            const TensorCPU& comm_ptr = Input(0);
            long ptr = *comm_ptr.template data<long>();

            int world_size, rank;
            MPI_Comm  comm = *((MPI_Comm *) ptr);
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &rank);

            auto* Y = Output(0);

            MPI_Status recv_status;

            MPI_Recv(Y->raw_mutable_data(), Y->nbytes(), MPIDataTypeWrapper<T>::type(),
            source_, tag_, comm, &recv_status);

            auto* src_out = OperatorBase::Output<TensorCPU>(1);
            src_out->Resize();
            src_out->template mutable_data<int>()[0] = recv_status.MPI_SOURCE;
            auto* tag_out = OperatorBase::Output<TensorCPU>(2);
            tag_out->Resize();
            tag_out->template mutable_data<int>()[0] = recv_status.MPI_TAG;

            return true;
        }


    protected:
        int source_;
        int tag_;
        Workspace* ws_;

    };

    template <typename T, class Context>
    class DMpiSendOp final : public Operator<CPUContext> {
    public:
        USE_OPERATOR_FUNCTIONS(CPUContext);

        DMpiSendOp(const OperatorDef& operator_def, Workspace* ws) : Operator<CPUContext>(operator_def, ws),
                                                                                OP_SINGLE_ARG(int, "dest", dest_, 0),
                                                                                OP_SINGLE_ARG(int, "tag", tag_, MPI_ANY_TAG),
                                                                                ws_(ws) {
        }


        bool RunOnDevice() override  {
            const TensorCPU& comm_ptr = Input(1);
            long ptr = *comm_ptr.template data<long>();

            int world_size, rank;
            MPI_Comm  comm = *((MPI_Comm *) ptr);
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &rank);


            auto& X = Input(0);

            MPI_Send(X.raw_data(), X.nbytes(), MPIDataTypeWrapper<T>::type(), dest_, tag_, comm);
            return true;
        }


    protected:
        int dest_;
        int tag_;
        Workspace* ws_;

    };


    REGISTER_CPU_OPERATOR(DMpiSend, DMpiSendOp<float, CPUContext>);
    REGISTER_CPU_OPERATOR(DMpiRecv, DMpiRecvOp<float, CPUContext>);
    REGISTER_CPU_OPERATOR(DMpiBroadcast, DMpiBroadcastOp<float, CPUContext>);
    REGISTER_CPU_OPERATOR(DMpiGather, DMpiGatherOp<float, CPUContext>);
    REGISTER_CPU_OPERATOR(DMpiAllReduce, DMpiAllreduceOp<float, CPUContext>);
    REGISTER_CPU_OPERATOR(DMpiAllReduceMean, DMpiAllreduceMeanOp<float, CPUContext>);
    REGISTER_CPU_OPERATOR(DMpiReduceMean, DMpiReduceMeanOp<float, CPUContext>);


    OPERATOR_SCHEMA(DMpiSend)
                    .NumInputs(2)
                    .Arg("dest", "destination to send the message (default root)", false)
                    .Arg("tag", "message tag (default any tag)", false)
                    .Input(0, "X", "The input X")
                    .Input(1, "mpi_comm", "mpi_comm");
    OPERATOR_SCHEMA(DMpiRecv)
                    .NumInputs(1).NumOutputs(3)
                    .Arg("source", "source from where to receive (default any source)", false)
                    .Arg("tag", "message tag (default any tag)", false)
                    .Input(0, "X", "The input X")
                    .Input(1, "mpi_comm", "mpi_comm")
                    .AllowInplace({{0,0}})
                    .Output(0, "Y", "The output")
                    .Output(1, "SOURCE", "Recv. from which source")
                    .Output(2, "TAG", "Tag of msg");
    OPERATOR_SCHEMA(DMpiBroadcast)
                    .NumInputs(2).NumOutputs(1)
                    .Input(0, "X", "The input X")
                    .Input(1, "mpi_comm", "mpi_comm")
                    .AllowInplace({{0,0}})
                    .Output(0, "Y", "The output");
    OPERATOR_SCHEMA(DMpiAllReduce)
                    .NumInputs(2).NumOutputs(1)
                    .Input(0, "X", "The input X")
                    .AllowInplace({{0,0}})
                    .Input(1, "mpi_comm", "mpi_comm")
                    .Output(0, "Y", "The output");
    OPERATOR_SCHEMA(DMpiAllReduceMean)
                    .NumInputs(2).NumOutputs(1)
                    .Input(0, "X", "The input X")
                    .AllowInplace({{0,0}})
                    .Input(1, "mpi_comm", "mpi_comm")
                    .Output(0, "Y", "The output");
    OPERATOR_SCHEMA(DMpiReduceMean)
                    .NumInputs(2).NumOutputs(1)
                    .Input(0, "X", "The input X")
                    .AllowInplace({{0,0}})
                    .Input(1, "mpi_comm", "mpi_comm")
                    .Output(0, "Y", "The output");
    OPERATOR_SCHEMA(DMpiGather)
                    .NumInputs(2).NumOutputs(1)
                    .Input(0, "X", "The input X")
                    .AllowInplace({{0,0}})
                    .Input(1, "mpi_comm", "mpi_comm")
                    .Output(0, "Y", "The output (x times the size of input where x is group size");

    SHOULD_NOT_DO_GRADIENT(DMpiSend);
    SHOULD_NOT_DO_GRADIENT(DMpiRecv);
    SHOULD_NOT_DO_GRADIENT(DMpiBroadcast);
    SHOULD_NOT_DO_GRADIENT(DMpiAllReduce);
    SHOULD_NOT_DO_GRADIENT(DMpiAllReduceMean);
    SHOULD_NOT_DO_GRADIENT(DMpiReduceMean);
    SHOULD_NOT_DO_GRADIENT(DMpiGather);


} // close namespace