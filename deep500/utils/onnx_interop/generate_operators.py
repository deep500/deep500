""" This file generates an object-oriented representation of ONNX operators for the Deep500 intermediate representation. """

from typing import List

from onnx import defs
from onnx.defs import OpSchema


class ClazzAttribute:
    def __init__(self, name, type, doc_string, optional=False, variadic=False):
        self.name = name
        self.type = type
        self.doc_string = doc_string
        self.optional = optional
        self.variadic = variadic

    @classmethod
    def create(cls, name, type, description):
        return cls(name, type, description)

class MethodToPrint:
    def __init__(self, name, attributes, decorators=[]):
        self.name = name
        self.attributes = attributes
        self.lines = []
        self.decorators = decorators

    def __str__(self):
        s = ""
        for each_decorator in self.decorators:
            s += " " * 4 + "@{}\n".format(each_decorator)
        s += " " * 4 + "def {}(".format(self.name)
        s += ', '.join(self.attributes)
        s += '):\n'

        if len(self.lines) == 0:
            s += "pass\n"
        else:
            for each_line in self.lines:
                s += " " * 8 + each_line + "\n"
        s += "\n"
        return s


class ClazzToPrint:
    def __init__(self, name, class_doc, attributes, inputs, outputs):
        self.name = name
        self.class_doc = class_doc
        self.attributes = [ClazzAttribute.create(attr_name, attributes[attr_name].type,
                                                 attributes[attr_name].description) for attr_name in attributes.keys()]
        self.inputs = [ClazzAttribute(inp.name, inp.typeStr, inp.description,
                                      OpSchema.FormalParameterOption.Optional == inp.option,
                                      OpSchema.FormalParameterOption.Variadic == inp.option) for inp in inputs]
        self.outputs = [ClazzAttribute(outs.name, outs.typeStr, outs.description,
                                       OpSchema.FormalParameterOption.Optional == outs.option,
                                       OpSchema.FormalParameterOption.Variadic == outs.option) for outs in outputs]

        constructor_args = ['input', 'output', 'name', 'op_type', 'domain', 'attributes', 'doc_string']

        # create init
        self.init_method = MethodToPrint("__init__", ['self'] + constructor_args)
        self.init_method.lines.append("super({}, self).__init__({})".format(self.name, ', '.join(constructor_args)))
        # add init attributes
        for attr in self.attributes:
            self.init_method.lines.append("self.{} = self.attributes.get('{}')".format(attr.name, attr.name))
        # add init inputs
        for i, inp in enumerate(self.inputs):
            self.init_method.lines.append("# {}".format(inp.doc_string))
            if inp.optional:
                self.init_method.lines.append('# OPTIONAL')
                self.init_method.lines.append(
                    "self.i_{} = None if len(self.input) < {} else self.input[{}]".format(inp.name, i + 1, i))
            elif inp.variadic:
                self.init_method.lines.append(
                    "# input is variadic [1,infty) just use self.input to access whole list")
            else:
                self.init_method.lines.append("self.i_{} = self.input[{}]".format(inp.name, i))

        # add init outputs
        for i, out in enumerate(self.outputs):
            self.init_method.lines.append("# {}".format(out.doc_string))
            if out.optional:
                self.init_method.lines.append('# OPTIONAL')
                self.init_method.lines.append(
                    "self.o_{} = None if len(self.output) < {} else self.output[{}]".format(out.name, i + 1, i))
            elif out.variadic:
                self.init_method.lines.append(
                    "# output is variadic [1,infty) just use self.output to access whole list")
            else:
                self.init_method.lines.append("self.o_{} = self.output[{}]".format(out.name, i))

        # add accept method
        self.accept_method = MethodToPrint("accept", ["self", "visitor", "network"])
        self.accept_method.lines.append("super({}, self).accept(visitor, network)".format(self.name))
        self.accept_method.lines.append("visitor.visit_{}(self, network)".format(self.name.lower()))

        self.create_op_method = self.get_create_op_method(self.inputs, self.outputs, self.attributes)

    def get_line(self, intend, text, arguments=None):
        if arguments is None:
            return " " * intend + text + "\n"
        return (" " * intend + text + "\n").format(arguments)

    def __str__(self):
        s = self.get_line(0, "class {}(Operation):", self.name)
        if self.class_doc is not None:
            s += self.get_line(4, "\"\"\"" + self.class_doc + "    \"\"\"\n")

        s += str(self.init_method)

        s += str(self.accept_method)

        s += str(self.create_op_method)

        s += "\n"

        return s

    def get_create_op_method(self,
                             inputs: List[ClazzAttribute],
                             outputs: List[ClazzAttribute],
                             attributes: List[ClazzAttribute]):
        inputs_ = []
        for each_input in inputs:
            inputs_.append("i_{}: {}".format(each_input.name, "str"))

        inputs_str = []
        for each_input in inputs:
            inputs_str.append("i_{}".format(each_input.name))

        outputs_ = []
        for each_output in outputs:
            outputs_.append("o_{}: {}".format(each_output.name, "str"))

        outputs_str = []
        for each_output in outputs:
            outputs_str.append("o_{}".format(each_output.name))

        attributes_ = []
        for each_attribute in attributes:
            attributes_.append("{}: {}".format(each_attribute.name, "OnnxAttribute"))

        # add create op method
        create_op_method = MethodToPrint('create_op', ['cls'] + inputs_ + outputs_ + attributes_, ['classmethod'])

        # create attributes hashmap
        create_op_method.lines.append("attributes = {")
        for each_attribute in attributes:
            create_op_method.lines.append(' ' * 4 + "'{}': {},".format(each_attribute.name, each_attribute.name))
        create_op_method.lines.append("}")
        #

        # create return
        inputs_str = ', '.join(inputs_str)
        outputs_str = ', '.join(outputs_str)
        create_op_method.lines.append("return cls([{}], [{}], None, None, None, attributes, None)".format(
            inputs_str, outputs_str))

        return create_op_method


def display_schema(schema):
    clazz = ClazzToPrint(schema.name, schema.doc, schema.attributes, schema.inputs, schema.outputs)
    return clazz


def main():
    clazzes = []
    for schema in defs.get_all_schemas():
        clazz = display_schema(schema)
        clazzes.append(clazz)

    with open('generated_operators.py', 'w') as code:
        code.truncate()  # empty the file

        # write header
        code.write('### THIS FILE IS AUTOMATICALLY GENERATED BY generate_operators.py, DO NOT MODIFY\n')
        
        # write imports
        code.write('from deep500.utils.onnx_interop.onnx_objects import OnnxAttribute, Operation\n\n\n')

        # write classes
        for each_class in clazzes:
            code.write(str(each_class))

        # write hashtable
        code.write("ONNX_OPERATIONS = {\n")
        for each_class in clazzes:
            code.write(" " * 4 + "'{}': {},\n".format(each_class.name.lower(), each_class.name))
        code.write("}\n")

    def write_visitor(raise_exception: bool, prefix: str):
        inner_text = "raise Exception('implement this method')\n\n" if raise_exception else "pass\n\n"
        with open('{}operations_visitor.py'.format(prefix.lower()), 'w') as visitor_file:
            visitor_file.truncate()

            # write import
            s = "import abc\n"
            s += "from deep500.lv1.network import Network\n"
            s += "from deep500.utils.onnx_interop.generated_operators import *\n\n\n"
            # write class
            s += 'class {}OperationsVisitor(abc.ABC):\n'.format(prefix)
            for each_class in clazzes:
                t = " " * 4 + "def visit_{}(self, op: {}, network: Network):\n".format(each_class.name.lower(),
                                                                                       each_class.name)
                t += " " * 8 + inner_text
                s += t
            visitor_file.write(s)

    write_visitor(True, "")
    write_visitor(False, "Empty")


if __name__ == '__main__':
    print("Genearting operator file")
    main()
