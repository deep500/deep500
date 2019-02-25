// Wrapper file for Tensorflow to compile custom Deep500 operations
#include <deep500/deep500.h>

#include <cstdio>
#include <cmath>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>


#define xstr(s) str(s)
#define str(s) #s
#define DSTR_OPNAME xstr({{opname}})
#define STR_CLASSNAME "TfOp{{opname}}"
#define STR_GCLASSNAME "TfOpGrad{{opname}}"

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

using namespace tensorflow;

// Register op
REGISTER_OP(STR_CLASSNAME)
.Attr("op_handle_ptr: int")
{% for type, tensor in input_tensors %}
.Input("{{tensor}}: {{type}}")
{% endfor %}
{% for type, tensor in output_tensors %}
.Output("{{tensor}}: {{type}}")
{% endfor %}
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    {% for type, tensor in output_tensors %}
        {% if output_shapes[loop.index0]|length == 0 %}
            c->set_output({{loop.index0}}, c->input({{loop.index0}}));
        {% else %}
            c->set_output({{loop.index0}}, c->MakeShape({
                    {% for s in output_shapes[loop.index0] %}
                        {{s}}{{ "," if not loop.last }}
                    {% endfor %}
            }));
        {% endif %}
    {% endfor %}
    return Status::OK();
});

// Register backward op
REGISTER_OP(STR_GCLASSNAME)
.Attr("op_handle_ptr: int")
{% for type, tensor in (nextop_grads + input_tensors + output_tensors) %}
.Input("{{tensor}}: {{type}}")
{% endfor %}
{% for type, tensor in input_grads %}
.Output("{{tensor}}: {{type}}")
{% endfor %}
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    {% for type, tensor in input_grads %}
        {% if input_shapes[loop.index0]|length == 0 %}
            c->set_output({{loop.index0}}, c->input({{loop.index0}}));
        {% else %}
            c->set_output({{loop.index0}}, c->MakeShape({
                    {% for s in input_shapes[loop.index0] %}
                        {{s}}{{ "," if not loop.last }}
                    {% endfor %}
            }));
        {% endif %}
    {% endfor %}
    return Status::OK();
});



{% for devicename, platform in platforms %}

#define CLASSNAME{{platform}} TfOp{{platform}}{{opname}}
#define GCLASSNAME{{platform}} TfOpGrad{{platform}}{{opname}}


// TODO(later): Use AsyncOpKernel?
class CLASSNAME{{platform}} : public OpKernel {
protected:
    void *m_handle;
public:
    explicit CLASSNAME{{platform}}(OpKernelConstruction* context) :
            OpKernel(context) {
        int64 handle;
        context->GetAttr("op_handle_ptr", &handle);
        m_handle = (void *)handle;
    }

    void Compute(OpKernelContext* context) override {
        // Obtain the input tensors
        {% for type, tensor in input_tensors %}
        const Tensor& {{tensor}}_t = context->input({{loop.index0}});
        const {{type}} *{{tensor}} = ({{type}}*){{tensor}}_t.tensor_data().data();
        {% endfor %}

        // Create output tensors
        {% for type, tensor in output_tensors %}
            {% set outer_loop = loop %}
            Tensor* output_{{loop.index0}} = nullptr;
            {% if output_shapes[loop.index0]|length == 0 %}
                OP_REQUIRES_OK(context, context->allocate_output({{loop.index0}}, context->input({{loop.index0}}).shape(), &output_{{loop.index0}}));
            {% else %}
                TensorShape output_shape_{{loop.index0}};
                {% for s in output_shapes[outer_loop.index0] %}
                    output_shape_{{outer_loop.index0}}.AddDim({{s}});
                {% endfor %}
                OP_REQUIRES_OK(context, context->allocate_output({{loop.index0}}, output_shape_{{loop.index0}}, &output_{{loop.index0}}));
            {% endif %}

            {{type}} *{{tensor}} = ({{type}}*)(output_{{loop.index0}}->tensor_data().data());
        {% endfor %}

        // Call custom op method
        _op_forward{{platform}}(m_handle,
            {% for type, tensor in (input_tensors + output_tensors) %}
            {{tensor}}{{ "," if not loop.last }}{% endfor %}
        );
    }
};

class GCLASSNAME{{platform}} : public OpKernel {
protected:
    void *m_handle;
public:
    explicit GCLASSNAME{{platform}}(OpKernelConstruction* context) :
            OpKernel(context) {
        int64 handle;
        context->GetAttr("op_handle_ptr", &handle);
        m_handle = (void *)handle;
    }

    void Compute(OpKernelContext* context) override {
        // Obtain the input tensors
        {% for type, tensor in (nextop_grads + input_tensors + output_tensors) %}
        const Tensor& {{tensor}}_t = context->input({{loop.index0}});
        const {{type}} *{{tensor}} = ({{type}}*){{tensor}}_t.tensor_data().data();
        {% endfor %}

        // Create output tensors
        {% for type, tensor in input_grads %}
            Tensor* output_{{loop.index0}} = nullptr;
            {% set outer_loop = loop %}
            {% if input_shapes[loop.index0]|length == 0 %}
                OP_REQUIRES_OK(context, context->allocate_output({{loop.index0}}, context->input({{loop.index0}}).shape(), &output_{{loop.index0}}));
            {% else %}
                TensorShape output_shape_{{loop.index0}};
                {% for s in input_shapes[outer_loop.index0] %}
                    output_shape_{{outer_loop.index0}}.AddDim({{s}});
                {% endfor %}
                OP_REQUIRES_OK(context, context->allocate_output({{loop.index0}}, output_shape_{{loop.index0}}, &output_{{loop.index0}}));
            {% endif %}

            {{type}} *{{tensor}} = ({{type}}*)(output_{{loop.index0}}->tensor_data().data());
        {% endfor %}

        // Call custom op method
        _op_backward{{platform}}(m_handle,
            {% for type, tensor in (nextop_grads + input_tensors + output_tensors + input_grads) %}
            {{tensor}}{{ "," if not loop.last }}{% endfor %}
        );
    }
};

// Forward
REGISTER_KERNEL_BUILDER(Name(STR_CLASSNAME).Device({{devicename}}), CLASSNAME{{platform}});
// Backward
REGISTER_KERNEL_BUILDER(Name(STR_GCLASSNAME).Device({{devicename}}), GCLASSNAME{{platform}});
{% endfor %}
