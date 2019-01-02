// Wrapper file for pytorch to compile custom Deep500 operations
#include <deep500/deep500.h>

#include <cstdio>
#include <cmath>

#include <TH/TH.h>

#define xstr(s) str(s)
#define str(s) #s
#define DSTR_OPNAME xstr(D500_OPNAME)

#include {{opfile}}

D500_EXPORTED bool is_cuda_supported(deep500::CustomOperator *handle)
{
    return handle->supports_cuda();
}

D500_EXPORTED int64_t report(deep500::CustomOperator *handle, void *data)
{
    return handle->report(data);
}

D500_EXPORTED void delete_op(deep500::CustomOperator *handle)
{
    delete handle;
}

{% for func_suffix in platforms %}

D500_EXPORTED void forward_op{{func_suffix}}(void *handle,
    {% for type, tensor in (input_tensors + output_tensors) %}
    TH{{type}}Tensor *{{tensor}}{{ "," if not loop.last }}{% endfor %})
{
    // Call object
    _op_forward{{func_suffix}}(handle,
        {% for type, tensor in (input_tensors + output_tensors) %}
        TH{{type}}Tensor_data({{tensor}}){{ "," if not loop.last }}{% endfor %}
    );
}

D500_EXPORTED void backward_op{{func_suffix}}(void *handle,
    {% for type, tensor in (nextop_grads + input_tensors +
                            output_tensors + input_grads) %}
    TH{{type}}Tensor *{{tensor}}{{ ", " if not loop.last }}{% endfor %}) 
{
    // Call object
    _op_backward{{func_suffix}}(handle,
        {% for type, tensor in (nextop_grads + input_tensors +
                            output_tensors + input_grads) %}
        TH{{type}}Tensor_data({{tensor}}){{ ", " if not loop.last }}{% endfor %}
    );
}

{% endfor %}
