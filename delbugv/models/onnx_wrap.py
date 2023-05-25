from __future__ import annotations

import onnx.numpy_helper
import onnxruntime
import numpy as np
import onnx
import os

from typing import List, Dict, Tuple, Sequence

from ..helper import duplicate_proto, merge_to_linearize, assertions_gemm_node_attributes, conv_to_gemm, assertions_conv_node_attributes, merge_to_combine, merge_to_combine_inputs



class ONNXWrap:
    """
    This class provides an interface to interact with an ONNX model. The class includes various properties and methods to extract information from the model, such as the input and output names, nodes, initializers, and variable shapes.

    Attributes:
        model (onnx.ModelProto): The loaded ONNX model.
        input_name (str): The name of the input tensor.
        output_name (str): The name of the output tensor.
        nodes (List[onnx.NodeProto]): The nodes in the ONNX model.
        output (onnx.ValueInfoProto): The output of the ONNX model.
        input (onnx.ValueInfoProto): The input of the ONNX model.
        node_idx_input (Dict[int, str]): A dictionary mapping node index to its input name.
        input_node_idx (Dict[str, int]): A dictionary mapping node input to its index in the nodes list.
        node_idx_output (Dict[int, str]): A dictionary mapping node index to its output name.
        output_node_idx (Dict[str, int]): A dictionary mapping node output to its index in the nodes list.
        initializers (List[onnx.TensorProto]): The list of initializers in the ONNX model.
        initializer_idx_value (Dict[int, np.ndarray]): A dictionary mapping initializer index to its value.
        initializer_name_idx (Dict[str, int]): A dictionary mapping initializer name to its index in initializers.
        shape (Dict[str, List[int]]): A dictionary mapping variable name to its shape.
        variables (List[str]): A list of the variables name in the model in order.
        inference_session (onnxruntime.InferenceSession): An InferenceSession to the model.

    Raises:
        TypeError: If the provided model is not an instance of onnx.ModelProto.
        ValueError: If the provided model is invalid or missing required metadata.
    """

    def __init__(self, model: onnx.ModelProto):
        """
        A wrapper around an ONNX model.

        Args:
            model (onnx.ModelProto): An loaded ONNX model object representing the model.

        Raises:
            TypeError: If the provided model is not an instance of onnx.ModelProto.
            ValueError: If the provided model is invalid or missing required metadata.
        """
        self.__model = model
        self.__input, self.__input_name = self.__get_input_and_name()
        self.__output, self.__output_name = self.__get_output_and_name()
        self.__nodes, self.node_name_idx = self.__get_nodes()
        self.__node_idx_input, self.__node_idx_initializer_names, self.__node_idx_output, self.__input_node_idx, self.__output_node_idx = self.__get_nodes_inputs_outputs()
        self.__initializers, self.__initializer_name_idx, self.__initializer_idx_value = self.__get_initializers_data()
        self.__variables, self.__shape = self.__get_variables_and_shapes()
        self.__inference_session = self.__get_inference_session()

    @property
    def model(self) -> onnx.ModelProto:
        """The loaded ONNX model."""
        return self.__model

    @property
    def input_name(self) -> str:
        """The name of the input tensor."""
        return self.__input_name

    @property
    def output_name(self) -> str:
        """The name of the output tensor."""
        return self.__output_name

    @property
    def nodes(self) -> List[onnx.NodeProto]:
        """The nodes in the ONNX model."""
        return self.__nodes

    @property
    def output(self) -> onnx.ValueInfoProto:
        """The output of the ONNX model."""
        return self.__output

    @property
    def input(self) -> onnx.ValueInfoProto:
        """The output of the ONNX model."""
        return self.__input

    @property
    def node_idx_initializer_names(self) -> Dict[int, str]:
        """A dictionary mapping node index to node input where the input is not an initializer."""
        return self.__node_idx_initializer_names

    @property
    def node_idx_input(self) -> Dict[int, str]:
        """A dictionary mapping node index to node input where the input is not an initializer."""
        return self.__node_idx_input

    @property
    def input_node_idx(self) -> Dict[str, int]:
        """A dictionary mapping node index to node input where the input is not an initializer."""
        return self.__input_node_idx

    @property
    def node_idx_output(self) -> Dict[int, str]:
        """A dictionary mapping node index to node output."""
        return self.__node_idx_output

    @property
    def output_node_idx(self) -> Dict[str, int]:
        """A dictionary mapping node index to node output."""
        return self.__output_node_idx

    @property
    def initializers(self) -> List[onnx.TensorProto]:
        """A dictionary mapping initializer index to initializer values."""
        return self.__initializers

    @property
    def initializer_idx_value(self) -> Dict[int, np.ndarray]:
        """A dictionary mapping initializer index to initializer values."""
        return self.__initializer_idx_value

    @property
    def initializer_name_idx(self) -> Dict[str, int]:
        """A dictionary mapping initializer name to its index in initializers."""
        return self.__initializer_name_idx

    @property
    def shape(self) -> Dict[str, List[int]]:
        """A dictionary mapping variable name to its shape."""
        return self.__shape

    @property
    def variables(self) -> List[str]:
        """A List of the variables names."""
        return self.__variables

    @property
    def inference_session(self) -> onnxruntime.InferenceSession:
        """An InferenceSession to the model."""
        return self.__inference_session

    def __get_input_and_name(self) -> onnx.ValueInfoProto:
        """Get the input tensor and its name from the ONNX model."""
        input_names = {i.name for i in self.model.graph.input}
        initializer_names = {i.name for i in self.model.graph.initializer}
        assert len(
            input_names - initializer_names) == 1,  "The ONNX model should have exactly one input."
        input_name = (input_names - initializer_names).pop()
        for i in self.model.graph.input:
            if i.name == input_name:
                return i, input_name
        raise ValueError(
            f"Input tensor {self.input_name} not found in ONNX model.")

    def __get_output_and_name(self) -> Tuple[onnx.ValueInfoProto, str]:
        """Get the output tensor and its name from the ONNX model."""
        assert len(
            self.model.graph.output) == 1, "The ONNX model should have exactly one output."
        return self.model.graph.output[0], self.model.graph.output[0].name

    def __get_nodes(self) -> Tuple[List[onnx.NodeProto], Dict[str, int]]:
        """Get a list of the nodes in the model and a dictionary mapping node names to their indices in the list."""
        nodes = []
        node_name_to_idx = {}
        nodes_idx_counter = 0
        prev_output = self.input_name
        prev_outputs = []
        for node in self.model.graph.node:
            # Check that the previous output is used as input in the current node.
            assert prev_output in node.input, f"The previous node output {prev_output} is not found in the current node input."
            # Check that the current node produces exactly one output.
            assert len(
                node.output) == 1, "The current node should have exactly one output."
            nodes.append(node)
            node_name_to_idx[node.name] = nodes_idx_counter
            nodes_idx_counter += 1
            current_node_output = node.output[0]
            # Check that the current node's output is not already an input in a previous node.
            assert current_node_output not in prev_outputs, f"The current node output {current_node_output} is already an output for a previous node."
            prev_outputs.append(prev_output)
            prev_output = current_node_output
        # Check that the last node produces the same output as specified in the ONNX model.
        assert self.output_name == prev_output
        return nodes, node_name_to_idx

    def __get_nodes_inputs_outputs(self) -> Tuple[Dict[int, str], Dict[int, List[str]], Dict[int, str], Dict[str, int], Dict[str, int]]:
        """Get dictionaries mapping node indices and input/output names:
            - A dictionary that maps a node index to its input name.
            - A dictionary that maps a node index to its initializers names.
            - A dictionary that maps a node index to its output name.
            - A dictionary that maps an input name to the index of the node that uses it.
            - A dictionary that maps an output name to the index of the node that produces it.
        """
        node_to_input = {}
        node_to_initializers = {}
        node_to_output = {}
        input_to_node = {}
        output_to_node = {}
        initializer_names = {
            init.name for init in self.model.graph.initializer}
        for node_idx, node in enumerate(self.nodes):
            init_inputs = [
                input_ for input_ in node.input if input_ in initializer_names]
            non_init_inputs = [
                input_ for input_ in node.input if input_ not in initializer_names]
            node_to_initializers[node_idx] = init_inputs
            # Check that the current node has exactly one non-initializer input.```
            assert len(
                non_init_inputs) == 1, "The current node should have exactly one non-initializer input."
            node_to_input[node_idx] = non_init_inputs[0]
            input_to_node[non_init_inputs[0]] = node_idx
            # Check that the current node produces exactly one output.
            assert len(
                node.output) == 1, "The current node should have exactly one output."
            node_to_output[node_idx] = node.output[0]
            output_to_node[node.output[0]] = node_idx
        return node_to_input, node_to_initializers, node_to_output, input_to_node, output_to_node

    def __get_initializers_data(self) -> Tuple[List[onnx.TensorProto], Dict[str, int], Dict[int, np.ndarray]]:
        """
        Get the values of the model's initializers:
            - A list of the initializers of the model.
            - A dictionary mapping each initializer's name to its index in the `initializers` list.
            - A dictionary mapping each initializer index to its value as a NumPy array.
        """
        initializer_to_value = {}
        name_to_initializer_idx = {}
        initializers = []
        for initializer_idx, initializer in enumerate(self.model.graph.initializer):
            initializers.append(initializer)
            # Check that the current node produces exactly one output.
            assert initializer.data_type == 1 or initializer.data_type == 7, "All initializers data_type should be 1."
            initializer_to_value[initializer_idx] = onnx.numpy_helper.to_array(
                initializer)
            name_to_initializer_idx[initializer.name] = initializer_idx
        return initializers, name_to_initializer_idx, initializer_to_value

    def __get_variables_and_shapes(self) -> Tuple[List[str], Dict[str, List[int]]]:
        """Gets a list of variable names in order and a dictionary containing the name and shape of each variable"""
        shape_dict = {}
        # Use shape_inference to get the shapes of the inner inputs and outputs
        model_with_shapes = onnx.shape_inference.infer_shapes(
            self.model, strict_mode=True)
        onnx.checker.check_model(model_with_shapes)

        # Iterate over all initializers and add their shapes to the shape_dict
        for initializer in model_with_shapes.graph.initializer:
            assert initializer.name not in shape_dict
            shape_dict[initializer.name] = [d for d in initializer.dims]

        # Add the input of the model and its shapes to the shape_dict
        assert self.input_name not in shape_dict, f"{self.input_name} already in shape_dict."
        shape_dict[self.input_name] = [
            d.dim_value for d in self.input.type.tensor_type.shape.dim]

        # Add the output of the model and its shapes to the shape_dict
        assert self.output_name not in shape_dict, f"{self.output_name} already in shape_dict."
        shape_dict[self.output_name] = [
            d.dim_value for d in self.output.type.tensor_type.shape.dim]

        # Iterate over all inferred shapes and add them to the shape_dict
        for value_info in model_with_shapes.graph.value_info:
            assert value_info.name not in shape_dict
            shape_dict[value_info.name] = [
                t.dim_value for t in value_info.type.tensor_type.shape.dim]

        # Prepare variables order list. It will hold the variables in order.
        variables_ordered = [self.input_name]

        # Iterate over all the nodes in the graph and verify that the input and output variables are in the shape_dict and add them to the variables_ordered list
        for node_idx, node in enumerate(self.nodes):
            input_ = self.node_idx_input[node_idx]
            assert input_ in shape_dict
            assert input_ in variables_ordered
            output_ = self.node_idx_output[node_idx]
            assert output_ in shape_dict
            variables_ordered.append(output_)
        assert self.output_name in variables_ordered

        return variables_ordered, shape_dict

    def __get_inference_session(self) -> onnxruntime.InferenceSession:
        """Get an ONNXRuntime InferenceSession for the ONNX model."""
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 2
        options.intra_op_num_threads = 1
        sess = onnxruntime.InferenceSession(onnx._serialize(self.model), providers=[
                                            'CPUExecutionProvider'], sess_options=options)
        # Test the InferenceSession.
        inputs_sample = np.random.uniform(
            low=-1.0, high=1.0, size=self.shape[self.input_name]).astype(np.float32)
        outputs = np.array(sess.run([self.output_name], {
                           self.input_name: inputs_sample})[0], dtype=np.float32)
        assert list(outputs.shape) == self.shape[self.output_name]
        return sess

    def __get_nodes_indices_of_op_type(self, op_type) -> List[int]:
        nodes_of_op_type = []
        for node_idx, node in enumerate(self.nodes):
            if node.op_type == op_type:
                nodes_of_op_type.append(node_idx)
        return nodes_of_op_type

    @classmethod
    def from_file(cls, path: str) -> ONNXWrap:
        """
        Load an ONNX model from a file and create an instance of ONNXWrap.

        Args:
            path: The path to the ONNX file.

        Returns:
            An instance of ONNXWrap with the loaded ONNX model.

        Raises:
            FileNotFoundError: If the specified path does not exist.
            IsADirectoryError: If the specified path is a directory and not a file.
            onnx.onnx_cpp2py_export.checker.ValidationError: If the ONNX model fails to pass validation.
        """
        model = onnx.load(path)
        return cls(model)

    def run(self, input_value: np.ndarray, input_name: str = None, output_name: str = None) -> np.ndarray:
        """
        Runs the ONNX model on the given input and returns the output. The input can be any input of any node (As if cutting the model from that node), ame for output.

        Args:
            input_value: A numpy array representing the input to the model.
            input_name: The name of an input to a node that will be used as the input of the model. If None, uses the default input of the model.
            output_name: The name of an output of a node that will be used as the output of the model. If None, uses the default output of the model.

        Returns:
            A numpy array representing the output of the model.

        Raises:
            AssertionError: If the input and output nodes are not in the correct order.
            AssertionError: If the output shape is not as expected.
        """
        if input_name is None:
            input_name = self.input_name
        if output_name is None:
            output_name = self.output_name

        input_value_rehaped = input_value.reshape( self.shape[input_name])
        
        if output_name == input_name:
            assert list(input_value_rehaped.shape) == self.shape[input_name], "input_value shape: "+ str(list(input_value.shape)) + ", expected shape: " + str(list(self.shape[input_name]))
            return input_value_rehaped
        
        assert self.variables.index(
            input_name) < self.variables.index(output_name)
        to_run_model = self
        if (self.input_name, self.output_name) != (input_name, output_name):
            to_run_model = self.cut(input_name, output_name)
        output_value = to_run_model.inference_session.run(
            [output_name], {input_name: input_value_rehaped})[0]
        assert list(output_value.shape) == to_run_model.shape[output_name]
        return np.array(output_value, dtype=np.float32)

    def run_random(self, input_name: str = None, output_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the ONNX model on randomly generated input and returns the input and output values.

        Args:
            input_name: The name of an input to a node that will be used as the input of the model. If None, uses the default input of the model.
            output_name: The name of an output of a node that will be used as the output of the model. If None, uses the default output of the model.

        Returns:
            A tuple containing:
                - A numpy array with the randomly generated input values.
                - A numpy array with the output values produced by the model.
        """
        if input_name is None:
            input_name = self.input_name
        if output_name is None:
            output_name = self.output_name

        inputs_sample = np.random.uniform(
            low=-50.0, high=50.0, size=self.shape[input_name]).astype(np.float32)
        outputs_sample = self.run(inputs_sample, input_name, output_name)
        assert list(outputs_sample.shape) == self.shape[output_name]
        return inputs_sample, outputs_sample

    def cut(self, new_input, new_output):
        # Check that new_input and new_output are variables in the variables list
        if new_input not in self.variables:
            raise ValueError(
                "new_input must be a variable in the variables list")
        if new_output not in self.variables:
            raise ValueError(
                "new_output must be a variable in the variables list")

        # Check that new_input appears before new_output in the variables list
        if self.variables.index(new_input) > self.variables.index(new_output):
            raise ValueError(
                "new_input must appear before new_output in the variables list")

        # Cut the beginning and ending of self.model
        start_node = self.model.get_node_by_output(new_input)
        end_node = self.model.get_node_by_input(new_output)
        self.model = self.model.extract_subgraph([start_node], [end_node])

        # Create a new model with new_input as its input and new_output as its output
        new_graph = onnx.helper.make_graph(self.model.nodes,
                                           "new_model",
                                           [onnx.helper.make_tensor_value_info(
                                               new_input, onnx.TensorProto.FLOAT, None)],
                                           [onnx.helper.make_tensor_value_info(new_output, onnx.TensorProto.FLOAT, None)])
        new_model = onnx.helper.make_model(new_graph)

        # Initialize the new model in a new ONNXWrap instance and return it
        return ONNXWrap(model=new_model)

    def cut(self, new_input, new_output) -> ONNXWrap:
        # Check that new_input and new_output are variables in the variables list
        if new_input not in self.variables:
            raise ValueError(
                "new_input must be a variable in the variables list")
        if new_output not in self.variables:
            raise ValueError(
                "new_output must be a variable in the variables list")

        # Check that new_input appears before new_output in the variables list
        if self.variables.index(new_input) >= self.variables.index(new_output):
            raise ValueError(
                "new_input must appear before new_output in the variables list")

        # Get the new input and output nodes' indices
        input_node_idx = self.input_node_idx[new_input]
        output_node_idx = self.output_node_idx[new_output]
        # Get the nodes between the input and output nodes
        required_nodes = self.nodes[input_node_idx:output_node_idx+1]
        initializers = []

        # Find the relevant initializers
        for node_idx in range(input_node_idx, output_node_idx+1):
            to_add_initializers_idx = []
            for initializer_name in self.node_idx_initializer_names[node_idx]:
                to_add_initializers_idx.append(self.initializer_name_idx[initializer_name])
            to_add_initializers_idx = sorted(to_add_initializers_idx)
            to_add_initializers = [self.initializers[idx] for idx in to_add_initializers_idx]
            initializers+=to_add_initializers
        return self.duplicate_partially(nodes=required_nodes, input_name=new_input, input_shape=self.shape[new_input], output_name=new_output, output_shape=self.shape[new_output], initializers=initializers)

    def duplicate(self) -> ONNXWrap:
        """Creates a deep copy of the ONNX model wrapped by the `ONNXWrap` instance and returns a new `ONNXWrap` object
        that wraps the duplicated model.

        Returns:
            A new `ONNXWrap` object that wraps a deep copy of the ONNX model.
        """
        duplicated_model = duplicate_proto(self.model)
        return ONNXWrap(duplicated_model)

    def copy(self) -> ONNXWrap:
        return self.duplicate()

    def duplicate_partially(self, graph_name: str = None, nodes: Sequence[onnx.NodeProto] = None, input_name: str = None, input_shape: Sequence[int] = None, output_name: str = None, output_shape: Sequence[int] = None, initializers: Sequence[onnx.TensorProto] = None, producer_name: str = None) -> ONNXWrap:
        """
        Creates a duplicate of the current ONNX model, but with some of its components optionally modified.

        Args:
            graph_name: The new name for the graph. If None, the graph name will be unchanged.
            nodes: The new list of nodes for the graph. If None, the list of nodes will be unchanged.
            input_name: The new name for the input tensor. If None, the input tensor name will be unchanged.
            input_shape: The new shape for the input tensor. Required if input_name is not None.
            output_name: The new name for the output tensor. If None, the output tensor name will be unchanged.
            output_shape: The new shape for the output tensor. Required if output_name is not None.
            initializers: The new list of initializers for the graph. If None, the list of initializers will be unchanged.
            producer_name: The new producer name for the model. If None, the producer name will be unchanged.

        Returns:
            A new instance of ONNXWrap representing the duplicated model.
        """
        duplicated_model = duplicate_proto(self.model)
        duplicated_graph = duplicated_model.graph

        are_initializers_in_input = len(duplicated_graph.input) != 1
        if are_initializers_in_input:
            assert duplicated_graph.input[-1].name == self.input_name, "The input ValueInfoProto must be the last in the model.graph.input."
        assert isinstance(duplicated_graph, onnx.GraphProto)
        if graph_name is not None:
            duplicated_graph.name = graph_name
        if nodes is not None:
            duplicated_graph.ClearField('node')
            duplicated_graph.node.extend(
                [duplicate_proto(node) for node in nodes])
        if initializers is not None:
            duplicated_graph.ClearField('initializer')
            duplicated_graph.initializer.extend(
                [duplicate_proto(initializer) for initializer in initializers])
        if input_name is not None:
            assert input_shape is not None, "If input_name is not None, input_shape must be not None as well."
            input_ = onnx.onnx.helper.make_tensor_value_info(
                input_name, onnx.TensorProto.FLOAT, list(input_shape))
            duplicated_graph.ClearField('input')
            duplicated_graph.input.extend([duplicate_proto(input_)])
        if output_name is not None:
            assert output_shape is not None, "If output_name is not None, output_shape must be not None as well."
            output_ = onnx.onnx.helper.make_tensor_value_info(
                output_name, onnx.TensorProto.FLOAT, list(output_shape))
            duplicated_graph.ClearField('output')
            duplicated_graph.output.extend([duplicate_proto(output_)])
        if producer_name is not None:
            duplicated_model.producer_name = producer_name

        if are_initializers_in_input and not (input_name is None and initializers is None):
            input_ = duplicated_graph.input[-1]
            initializers_in_input = []
            for initializer in duplicated_graph.initializer:
                initializer_shape = list(
                    onnx.numpy_helper.to_array(initializer).shape)
                initializer_tensor = onnx.onnx.helper.make_tensor_value_info(
                    initializer.name, onnx.TensorProto.FLOAT, initializer_shape)
                initializers_in_input.append(initializer_tensor)
            inputs = initializers_in_input + [input_]
            duplicated_graph.ClearField('input')
            duplicated_graph.input.extend(inputs)

        onnx.checker.check_model(duplicated_model)
        new_onnx_wrap = ONNXWrap(duplicated_model)
        return new_onnx_wrap

    @classmethod
    def create(cls, nodes: Sequence[onnx.NodeProto], graph_name: str, input_name: str, input_shape: Sequence[int], output_name: str, output_shape: Sequence[int], initializers: Sequence[onnx.TensorProto], producer_name: str) -> ONNXWrap:
        """
        Create a new ONNX model with the given nodes, graph name, input and output information, initializers, and producer name.

        Args:
            nodes (Sequence[onnx.NodeProto]): The list of ONNX nodes that make up the model.
            graph_name (str): The name of the graph in the ONNX model.
            input_name (str): The name of the input tensor in the ONNX model.
            input_shape (Sequence[int]): The shape of the input tensor as a list of integers.
            output_name (str): The name of the output tensor in the ONNX model.
            output_shape (Sequence[int]): The shape of the output tensor as a list of integers.
            initializers (Sequence[onnx.TensorProto]): The list of initializers for the model.
            producer_name (str): The name of the producer of the ONNX model.

        Returns:
            ONNXWrap: An ONNXWrap object representing the new ONNX model.
        """
        input_tensor = onnx.onnx.helper.make_tensor_value_info(
            input_name, onnx.TensorProto.FLOAT, list(input_shape))
        output_tensor = onnx.onnx.helper.make_tensor_value_info(
            output_name, onnx.TensorProto.FLOAT, list(output_shape))

        new_graph = onnx.onnx.helper.make_graph(nodes, graph_name, [input_tensor], [
                                                output_tensor], initializer=initializers)
        new_model = onnx.onnx.helper.make_model(
            new_graph, producer_name=producer_name)
        onnx.checker.check_model(new_model)
        return cls(new_model)
    
    def count_neurons(self):
        total_neurons = 0
        total_neurons += np.prod(self.shape[self.input_name])
        total_neurons += np.prod(self.shape[self.output_name])
        for node_idx, node in enumerate(self.nodes):
            if node.op_type == "Relu":
                shape = self.shape[self.node_idx_output[node_idx]]
                total_neurons += np.prod(shape)
        return total_neurons

    def concatenate(self, other: ONNXWrap) -> ONNXWrap:
        return concatenate(self, other)

    def get_linearizable_sequences(self) -> List[Triple[int, int]]:
        sequences = []
        relu_indices = self.__get_nodes_indices_of_op_type('Relu')
        for relu_idx in relu_indices:
            nodes_op_types = [
                node.op_type for node in self.nodes[relu_idx-1: relu_idx+2]]
            if nodes_op_types == ['Add', 'Relu', 'MatMul']:
                assert self.nodes[relu_idx-2].op_type == "MatMul"
                assert self.nodes[relu_idx+2].op_type == "Add"
                sequences.append(((relu_idx-2, relu_idx+3), ['Add', 'Relu', 'MatMul']))
            elif nodes_op_types == ['Gemm', 'Relu', 'Gemm']:
                sequences.append(((relu_idx-1, relu_idx+2), ['Gemm', 'Relu', 'Gemm']))
            elif nodes_op_types == ['Conv', 'Relu', 'Flatten']:
                assert self.nodes[relu_idx+2].op_type == "Gemm"
                sequences.append(((relu_idx-1, relu_idx+3), ['Conv', 'Relu', 'Flatten']))
            elif nodes_op_types == ['Conv', 'Relu', 'Conv']:
                sequences.append(((relu_idx-1, relu_idx+2), ['Conv', 'Relu', 'Conv']))
            else:
                raise ValueError("Unknown sequence: " + str(nodes_op_types))
        return sequences

    def get_combinable_pairs(self, assignment: np.ndarray) -> Dict[Triple[Triple[int, int], Triple[int, int]], float]:
        
        combinable_pairs_dict =   self.get_combinable_neuron_pairs_values(assignment)
        sorted_combinable_pairs = {k: v for k, v in sorted(
            combinable_pairs_dict.items(), key=lambda item: item[1][0])}
        sorted_combinable_pairs = {k: v[0] for k, v in sorted(
            sorted_combinable_pairs.items(), key=lambda item: item[1][1])}
        return sorted_combinable_pairs

    def get_combinable_neuron_pairs_values(self, assignment: np.ndarray) -> Dict[Triple[Triple[int, int], Triple[int, int]], Triple[float, float]]:
        
    # find combinable pairs
        combinable_pairs = []
        relu_indices = self.__get_nodes_indices_of_op_type('Relu')
        for relu_idx in relu_indices:
            nodes_op_types = [
                node.op_type for node in self.nodes[relu_idx-1: relu_idx+2]]
            if nodes_op_types == ['Add', 'Relu', 'MatMul']:
                assert self.nodes[relu_idx-2].op_type == "MatMul"
                assert self.nodes[relu_idx+2].op_type == "Add"
                combinable_pairs.append((relu_idx-2, relu_idx, relu_idx+2))
            elif nodes_op_types == ['Conv', 'Relu', 'Conv']:
                pass
            elif nodes_op_types == ['Gemm', 'Relu', 'Gemm']:
                combinable_pairs.append((relu_idx-1, relu_idx,  relu_idx+2))
            elif nodes_op_types == ['Conv', 'Relu', 'Flatten']:
                pass
            else:
                raise ValueError("Unknown sequence: " + str(nodes_op_types))
        combinable_pairs_dict = {}

        # sort the pairs 
        for fc_1_idx, relu_idx, fc_2_idx in combinable_pairs:
            model_cut_to_relu = self.cut(
                self.input_name, self.node_idx_input[relu_idx])
            before_relu_values = model_cut_to_relu.run(assignment)
            assert np.prod(before_relu_values.shape) == np.max(
                before_relu_values.shape)
            before_relu_values = before_relu_values.flatten()
            for i in range(before_relu_values.shape[0]):
                for j in range(i+1, before_relu_values.shape[0]):
                    diff = np.abs(
                        before_relu_values[i] - before_relu_values[j])
                    status = 0
                    if before_relu_values[i] == 0 and before_relu_values[j] == 0:
                        status = 1
                    elif before_relu_values[i] <= 0 and before_relu_values[j] <= 0:
                        status = 2
                    elif before_relu_values[i] >= 0 and before_relu_values[j] >= 0:
                        status = 3
                    else:
                        assert before_relu_values[i] > 0 and before_relu_values[
                            j] < 0 or before_relu_values[i] < 0 and before_relu_values[j] > 0
                        status = 4
                    assert status > 0
                    combinable_pairs_dict[(
                        (fc_1_idx, fc_2_idx), (i, j))] = before_relu_values[i], before_relu_values[j]
        input_shape = self.shape[self.input_name]
        if np.max(input_shape) == np.prod(input_shape):
            for i in range(np.prod(input_shape)):
                for j in range(i+1, np.prod(input_shape)):  
                    combinable_pairs_dict[(
                        (0, relu_indices[0]), (i, j))] = assignment.flatten()[i], assignment.flatten()[j]
                      
                    
        return combinable_pairs_dict

    def linearize(self, first_node_idx: int, second_node_idx: int, assignment: np.ndarray) -> ONNXWrap:
        """
        Linearizes a subgraph of the model between two nodes.

        Args:
            first_node_idx (int): The index of the first node in the subgraph.
            second_node_idx (int): The index of the node immediately following the end of the subgraph.
            assignment (np.ndarray): The input values for the subgraph.

        Returns:
            ONNXWrap: A wrapped ONNX model that represents the linearized subgraph.

        Raises:
            AssertionError: If the operation types of the nodes in the subgraph do not match one of the known linearization patterns: 'MatMul', 'Add', 'Relu', 'MatMul', 'Add' or 'Gemm', 'Relu', 'Gemm' or 'Conv', 'Relu', 'Flatten', 'Gemm' or 'Conv', 'Relu', 'Conv'.
        """
        nodes_op_types = [
            node.op_type for node in self.nodes[first_node_idx: second_node_idx]]
        if nodes_op_types == ['MatMul', 'Add', 'Relu', 'MatMul', 'Add']:
            before_relu_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx+2]).flatten()
            activated_neurons = np.where(before_relu_values >= 0, 1, 0)

            conv_1_weights_name = self.node_idx_initializer_names[first_node_idx][0]
            conv_1_biases_name = self.node_idx_initializer_names[first_node_idx+1][0]
            conv_2_weights_name = self.node_idx_initializer_names[first_node_idx+3][0]
            conv_2_biases_name = self.node_idx_initializer_names[first_node_idx+4][0]

            conv_1_weights_idx = self.initializer_name_idx[conv_1_weights_name]
            conv_1_biases_idx = self.initializer_name_idx[conv_1_biases_name]
            conv_2_weights_idx = self.initializer_name_idx[conv_2_weights_name]
            conv_2_biases_idx = self.initializer_name_idx[conv_2_biases_name]

            assert conv_1_weights_idx + 1 == conv_1_biases_idx and conv_1_biases_idx + \
                1 == conv_2_weights_idx and conv_2_weights_idx+1 == conv_2_biases_idx
            first_initializer_idx, last_initializer_idx = conv_1_weights_idx, conv_2_biases_idx+1

            conv_1_weights = self.initializer_idx_value[conv_1_weights_idx]
            conv_1_biases = self.initializer_idx_value[conv_1_biases_idx]
            conv_2_weights = self.initializer_idx_value[conv_2_weights_idx]
            conv_2_biases = self.initializer_idx_value[conv_2_biases_idx]

            new_weights, new_biases = merge_to_linearize(
                conv_1_weights, conv_1_biases, activated_neurons, conv_2_weights, conv_2_biases)
            new_matmul_node = duplicate_proto(self.nodes[first_node_idx])
            new_add_node = duplicate_proto(self.nodes[first_node_idx+1])
            new_add_node.output.pop()
            new_add_node.output.insert(
                0, self.node_idx_output[second_node_idx-1])
            new_weights_initializer = onnx.numpy_helper.from_array(
                new_weights, conv_1_weights_name)
            new_biases_initializer = onnx.numpy_helper.from_array(
                new_biases, conv_1_biases_name)
            new_nodes = [new_matmul_node, new_add_node]
            new_initializers = [
                new_weights_initializer, new_biases_initializer]

        elif nodes_op_types == ['Gemm', 'Relu', 'Gemm']:
            before_relu_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx+1]).flatten()
            assertions_gemm_node_attributes(self.nodes[first_node_idx])
            assertions_gemm_node_attributes(self.nodes[first_node_idx+2])
            activated_neurons = np.where(before_relu_values >= 0, 1, 0)

            conv_1_weights_name = self.node_idx_initializer_names[first_node_idx][0]
            conv_1_biases_name = self.node_idx_initializer_names[first_node_idx][1]
            conv_2_weights_name = self.node_idx_initializer_names[first_node_idx+2][0]
            conv_2_biases_name = self.node_idx_initializer_names[first_node_idx+2][1]

            conv_1_weights_idx = self.initializer_name_idx[conv_1_weights_name]
            conv_1_biases_idx = self.initializer_name_idx[conv_1_biases_name]
            conv_2_weights_idx = self.initializer_name_idx[conv_2_weights_name]
            conv_2_biases_idx = self.initializer_name_idx[conv_2_biases_name]

            assert conv_1_biases_idx + 1 == conv_1_weights_idx and conv_1_weights_idx + \
                1 == conv_2_biases_idx and conv_2_biases_idx+1 == conv_2_weights_idx
            first_initializer_idx, last_initializer_idx = conv_1_biases_idx, conv_2_weights_idx+1

            conv_1_weights = self.initializer_idx_value[conv_1_weights_idx].T
            conv_1_biases = self.initializer_idx_value[conv_1_biases_idx]
            conv_2_weights = self.initializer_idx_value[conv_2_weights_idx].T
            conv_2_biases = self.initializer_idx_value[conv_2_biases_idx]

            new_weights, new_biases = merge_to_linearize(
                conv_1_weights, conv_1_biases, activated_neurons, conv_2_weights, conv_2_biases)
            new_weights = new_weights.T

            new_gemm_node = duplicate_proto(self.nodes[first_node_idx])
            new_gemm_node.output.pop()
            new_gemm_node.output.insert(
                0, self.node_idx_output[second_node_idx-1])

            new_weights_initializer = onnx.numpy_helper.from_array(
                new_weights, conv_1_weights_name)
            new_biases_initializer = onnx.numpy_helper.from_array(
                new_biases, conv_1_biases_name)

            new_nodes = [new_gemm_node]
            new_initializers = [
                new_biases_initializer, new_weights_initializer]

        elif nodes_op_types == ['Conv', 'Relu', 'Flatten', 'Gemm']:
            before_relu_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx+1]).flatten()
            assertions_gemm_node_attributes(self.nodes[first_node_idx+3])
            activated_neurons = np.where(before_relu_values >= 0, 1, 0)

            conv_1_input_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx]).flatten()

            conv_1_weights_name = self.node_idx_initializer_names[first_node_idx][0]
            conv_1_biases_name = self.node_idx_initializer_names[first_node_idx][1]
            conv_2_weights_name = self.node_idx_initializer_names[first_node_idx+3][0]
            conv_2_biases_name = self.node_idx_initializer_names[first_node_idx+3][1]

            conv_1_weights_idx = self.initializer_name_idx[conv_1_weights_name]
            conv_1_biases_idx = self.initializer_name_idx[conv_1_biases_name]
            conv_2_weights_idx = self.initializer_name_idx[conv_2_weights_name]
            conv_2_biases_idx = self.initializer_name_idx[conv_2_biases_name]

            conv_1_weights = self.initializer_idx_value[conv_1_weights_idx]
            conv_1_biases = self.initializer_idx_value[conv_1_biases_idx]
            conv_2_weights = self.initializer_idx_value[conv_2_weights_idx].T
            conv_2_biases = self.initializer_idx_value[conv_2_biases_idx]

            conv_1_alternative_weights, conv_1_alternative_biases = conv_to_gemm(self.nodes[first_node_idx],
                                                                                 self.shape[self.node_idx_input[first_node_idx]], conv_1_weights, conv_1_biases, conv_1_input_values)

            assert conv_1_biases_idx + 1 == conv_1_weights_idx and conv_1_weights_idx + \
                1 == conv_2_biases_idx and conv_2_biases_idx+1 == conv_2_weights_idx
            first_initializer_idx, last_initializer_idx = conv_1_biases_idx, conv_2_weights_idx+1

            new_weights, new_biases = merge_to_linearize(
                conv_1_alternative_weights, conv_1_alternative_biases, activated_neurons, conv_2_weights, conv_2_biases)
            new_weights = new_weights.T

            conv_input_name = self.node_idx_input[first_node_idx]
            new_flatten_node = duplicate_proto(self.nodes[first_node_idx+2])
            new_flatten_node.input.pop(0)
            new_flatten_node.input.insert(0, conv_input_name)

            new_gemm_node = duplicate_proto(self.nodes[first_node_idx+3])

            new_weights_initializer = onnx.numpy_helper.from_array(
                new_weights, conv_2_weights_name)
            new_biases_initializer = onnx.numpy_helper.from_array(
                new_biases, conv_2_biases_name)

            new_nodes = [new_flatten_node, new_gemm_node]
            new_initializers = [new_biases_initializer, new_weights_initializer]

        elif nodes_op_types == ['Conv', 'Relu', 'Conv']:

            conv_1_input = self.node_idx_input[first_node_idx]
            conv_1_output = relu_input = self.node_idx_input[first_node_idx+1]
            relu_output = conv_2_input = self.node_idx_input[first_node_idx+2]
            conv_2_output = self.node_idx_output[first_node_idx+2]

            assertions_conv_node_attributes(self.nodes[first_node_idx])
            assertions_conv_node_attributes(self.nodes[first_node_idx+2])
            before_relu_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx+1]).flatten()
            activated_neurons = np.where(before_relu_values >= 0, 1, 0)

            conv_1_input_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx]).flatten()
            conv_2_input_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx+2]).flatten()

            conv_1_weights_name = self.node_idx_initializer_names[first_node_idx][0]
            conv_1_biases_name = self.node_idx_initializer_names[first_node_idx][1]
            conv_2_weights_name = self.node_idx_initializer_names[first_node_idx+2][0]
            conv_2_biases_name = self.node_idx_initializer_names[first_node_idx+2][1]

            conv_1_weights_idx = self.initializer_name_idx[conv_1_weights_name]
            conv_1_biases_idx = self.initializer_name_idx[conv_1_biases_name]
            conv_2_weights_idx = self.initializer_name_idx[conv_2_weights_name]
            conv_2_biases_idx = self.initializer_name_idx[conv_2_biases_name]

            conv_1_weights = self.initializer_idx_value[conv_1_weights_idx]
            conv_1_biases = self.initializer_idx_value[conv_1_biases_idx]
            conv_2_weights = self.initializer_idx_value[conv_2_weights_idx]
            conv_2_biases = self.initializer_idx_value[conv_2_biases_idx]

            conv_1_alternative_weights, conv_1_alternative_biases = conv_to_gemm(self.nodes[first_node_idx],
                                                                                 self.shape[self.node_idx_input[first_node_idx]], conv_1_weights, conv_1_biases, conv_1_input_values)

            conv_2_alternative_weights, conv_2_alternative_biases = conv_to_gemm(self.nodes[first_node_idx+2],
                                                                                 self.shape[self.node_idx_input[first_node_idx+2]], conv_2_weights, conv_2_biases, conv_2_input_values)

            assert conv_1_biases_idx + 1 == conv_1_weights_idx and conv_1_weights_idx + \
                1 == conv_2_biases_idx and conv_2_biases_idx+1 == conv_2_weights_idx
            first_initializer_idx, last_initializer_idx = conv_1_biases_idx, conv_2_weights_idx+1

            new_weights, new_biases = merge_to_linearize(
                conv_1_alternative_weights, conv_1_alternative_biases, activated_neurons, conv_2_alternative_weights, conv_2_alternative_biases)
            new_weights = new_weights.T

            conv_2_output_name = self.node_idx_output[first_node_idx+2]

            flatten_node = onnx.helper.make_node(
                "Flatten",
                inputs=[conv_1_input],
                outputs=[conv_1_output],
                axis=1,
            )
            gemm_node = onnx.helper.make_node(
                "Gemm",
                inputs=[relu_input, conv_1_weights_name, conv_1_biases_name],
                outputs=[relu_output],
                alpha=1.0,
                beta=1.0,
                transB=1,
            )
            reshape_node = onnx.helper.make_node(
                "Reshape",
                inputs=[conv_2_input, conv_2_input + "_shape"],
                outputs=[conv_2_output]
            )

            new_weights_initializer = onnx.numpy_helper.from_array(
                new_weights, conv_1_weights_name)
            new_biases_initializer = onnx.numpy_helper.from_array(
                new_biases, conv_1_biases_name)
            new_shape_initializer = onnx.numpy_helper.from_array(
                np.array(self.shape[conv_2_output]).astype(int), conv_2_input + "_shape")

            new_nodes = [flatten_node, gemm_node, reshape_node]
            new_initializers = [
                new_biases_initializer, new_weights_initializer, new_shape_initializer]
        else:
            raise ValueError(
                f"Unknown sequence of nodes' types before and after the node.")

        new_nodes = self.nodes[:first_node_idx] + \
            new_nodes + self.nodes[second_node_idx:]
        new_initializers = self.initializers[:first_initializer_idx] + \
            new_initializers + self.initializers[last_initializer_idx:]

        return self.duplicate_partially(nodes=new_nodes, initializers=new_initializers)

    def combine(self, nodes_idx: Triple[int, int], neurons_idx: Triple[int, int], assignment: np.ndarray) -> ONNXWrap:
        """
        Combine two neurons into one.

        Args:
            first_node_idx (int): The index of the first fully-connected node.
            second_node_idx (int): The index of the fully-connected node immediately following the first.
            assignment (np.ndarray): The input values for the subgraph.

        Returns:
            ONNXWrap: A wrapped ONNX model that represents the linearized subgraph.

        Raises:
            AssertionError: If the operation types of the nodes in the subgraph do not match one of the known linearization patterns: 'MatMul', 'Add', 'Relu', 'MatMul', 'Add' or 'Gemm', 'Relu', 'Gemm' or 'Conv', 'Relu', 'Flatten', 'Gemm' or 'Conv', 'Relu', 'Conv'.
        """
        first_node_idx, second_node_idx = nodes_idx[0], nodes_idx[1]
        first_neuron_idx, second_neuron_idx = neurons_idx[0], neurons_idx[1]
        assert first_neuron_idx < second_neuron_idx
        nodes_op_types = [
            node.op_type for node in self.nodes[first_node_idx: second_node_idx]]
        input_shape = None
        input_name = None
        if nodes_op_types == ['MatMul', 'Add', 'Relu', 'MatMul']:
            
            before_matmul1_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx]).flatten()
            before_add1_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx+1]).flatten()
            before_relu_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx+2]).flatten()
            
            after_relu_values = self.run(
                assignment, output_name=self.node_idx_output[first_node_idx+2]).flatten()
            after_matmul_values = self.run(
                assignment, output_name=self.node_idx_output[first_node_idx+3]).flatten()
            
            neuron_1_value = before_relu_values[first_neuron_idx]
            neuron_2_value = before_relu_values[second_neuron_idx]

            matmul_1_weights_name = self.node_idx_initializer_names[first_node_idx][0]
            matmul_1_biases_name = self.node_idx_initializer_names[first_node_idx+1][0]
            matmul_2_weights_name = self.node_idx_initializer_names[first_node_idx+3][0]

            matmul_1_weights_idx = self.initializer_name_idx[matmul_1_weights_name]
            matmul_1_biases_idx = self.initializer_name_idx[matmul_1_biases_name]
            matmul_2_weights_idx = self.initializer_name_idx[matmul_2_weights_name]

            assert matmul_1_weights_idx + 1 == matmul_1_biases_idx and matmul_1_biases_idx +  1 == matmul_2_weights_idx 
            first_initializer_idx, last_initializer_idx = matmul_1_weights_idx, matmul_2_weights_idx+1

            matmul_1_weights = self.initializer_idx_value[matmul_1_weights_idx]
            matmul_1_biases = self.initializer_idx_value[matmul_1_biases_idx]
            matmul_2_weights = self.initializer_idx_value[matmul_2_weights_idx]


            new_weights_1, new_biases_1, new_weights_2 = merge_to_combine(
                matmul_1_weights, matmul_1_biases, matmul_2_weights, first_neuron_idx, second_neuron_idx, before_relu_values)
            new_matmul_1_node = duplicate_proto(self.nodes[first_node_idx])
            new_add_1_node = duplicate_proto(self.nodes[first_node_idx+1])
            new_relu_node = duplicate_proto(self.nodes[first_node_idx+2])
            new_matmul_2_node = duplicate_proto(self.nodes[first_node_idx+3])
            
            new_weights_1_initializer = onnx.numpy_helper.from_array(
                new_weights_1, matmul_1_weights_name)
            new_biases_1_initializer = onnx.numpy_helper.from_array(
                new_biases_1, matmul_1_biases_name)
            new_weights_2_initializer = onnx.numpy_helper.from_array(
                new_weights_2, matmul_2_weights_name)
            new_nodes = [new_matmul_1_node, new_add_1_node, new_relu_node, new_matmul_2_node]
            new_initializers = [
                new_weights_1_initializer, new_biases_1_initializer, new_weights_2_initializer]

        elif nodes_op_types == ['Gemm', 'Relu', 'Gemm']:

            before_relu_values = self.run(
                assignment, output_name=self.node_idx_input[first_node_idx+1]).flatten()
            assertions_gemm_node_attributes(self.nodes[first_node_idx])
            assertions_gemm_node_attributes(self.nodes[first_node_idx+2])
            activated_neurons = np.where(before_relu_values >= 0, 1, 0)

            gemm_1_weights_name = self.node_idx_initializer_names[first_node_idx][0]
            gemm_1_biases_name = self.node_idx_initializer_names[first_node_idx][1]
            gemm_2_weights_name = self.node_idx_initializer_names[first_node_idx+2][0]
            gemm_2_biases_name = self.node_idx_initializer_names[first_node_idx+2][1]

            gemm_1_weights_idx = self.initializer_name_idx[gemm_1_weights_name]
            gemm_1_biases_idx = self.initializer_name_idx[gemm_1_biases_name]
            gemm_2_weights_idx = self.initializer_name_idx[gemm_2_weights_name]
            gemm_2_biases_idx = self.initializer_name_idx[gemm_2_biases_name]

            assert gemm_1_biases_idx + 1 == gemm_1_weights_idx and gemm_1_weights_idx + \
                1 == gemm_2_biases_idx and gemm_2_biases_idx+1 == gemm_2_weights_idx
            first_initializer_idx, last_initializer_idx = gemm_1_biases_idx, gemm_2_weights_idx+1

            gemm_1_weights = self.initializer_idx_value[gemm_1_weights_idx].T
            gemm_1_biases = self.initializer_idx_value[gemm_1_biases_idx]
            gemm_2_weights = self.initializer_idx_value[gemm_2_weights_idx].T
            gemm_2_biases = self.initializer_idx_value[gemm_2_biases_idx]

            new_weights_1, new_biases_1, new_weights_2 = merge_to_combine(
                gemm_1_weights, gemm_1_biases, gemm_2_weights, first_neuron_idx, second_neuron_idx, before_relu_values)

            new_gemm_1_node = duplicate_proto(self.nodes[first_node_idx])
            new_relu_node = duplicate_proto(self.nodes[first_node_idx+1])
            new_gemm_2_node = duplicate_proto(self.nodes[first_node_idx+2])
            
            new_weights_1 = new_weights_1.T
            new_weights_2 = new_weights_2.T
            new_biases_2 = np.array(gemm_2_biases)

            new_weights_1_initializer = onnx.numpy_helper.from_array(
                new_weights_1, gemm_1_weights_name)
            new_biases_1_initializer = onnx.numpy_helper.from_array(
                new_biases_1, gemm_1_biases_name)
            new_weights_2_initializer = onnx.numpy_helper.from_array(
                new_weights_2, gemm_2_weights_name)
            new_biases_2_initializer = onnx.numpy_helper.from_array(
                new_biases_2, gemm_2_biases_name)
            new_nodes = [new_gemm_1_node, new_relu_node, new_gemm_2_node]
            new_initializers = [new_biases_1_initializer, new_weights_1_initializer, new_biases_2_initializer, new_weights_2_initializer]
        
        elif nodes_op_types == ['Flatten', 'Gemm']:
            assert first_node_idx == 0

            gemm_1_weights_name = self.node_idx_initializer_names[first_node_idx+1][0]
            gemm_1_weights_idx = self.initializer_name_idx[gemm_1_weights_name]

            assert gemm_1_weights_idx == 1
            first_initializer_idx, last_initializer_idx = gemm_1_weights_idx, gemm_1_weights_idx+1
            
            gemm_1_weights = self.initializer_idx_value[gemm_1_weights_idx].T
            new_weights_1 = merge_to_combine_inputs(gemm_1_weights, first_neuron_idx, second_neuron_idx, assignment)
            
            new_flatten_node = duplicate_proto(self.nodes[first_node_idx])
            new_gemm_1_node = duplicate_proto(self.nodes[first_node_idx+1])
            
            new_weights_1 = new_weights_1.T
            new_weights_1_initializer = onnx.numpy_helper.from_array(
                new_weights_1, gemm_1_weights_name)
            
            new_nodes = [new_flatten_node, new_gemm_1_node]
            new_initializers = [new_weights_1_initializer]
            
            input_shape = self.shape[self.input_name]
            max_idx = np.argmax(input_shape)
            input_shape[max_idx]-=1
            input_name = self.input_name
        

        else:
            raise ValueError(
                f"Unknown sequence of nodes' types before and after the node: "+str(nodes_op_types))

        new_nodes = self.nodes[:first_node_idx] + \
            new_nodes + self.nodes[second_node_idx:]
        new_initializers = self.initializers[:first_initializer_idx] + \
            new_initializers + self.initializers[last_initializer_idx:]
        return self.duplicate_partially(nodes=new_nodes, initializers=new_initializers, input_name=input_name, input_shape=input_shape)
    
    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        onnx.save(self.model, filename)


    def delete_flatten(self, assignment) -> ONNXWrap:
        assert self.nodes[0].op_type == "Flatten"
        assert self.nodes[1].op_type == "Gemm"
        
        return self.cut(self.node_idx_input[1], self.output_name)

    def flatten_input(self, assignment) -> ONNXWrap:
        assert self.nodes[0].op_type == "Flatten"
        assert self.nodes[1].op_type == "Gemm"
        
        input_shape = self.shape[self.input_name]
        input_shape = np.array([1, np.prod(input_shape)]).astype(int)
        input_shape = [int(x) for x in input_shape]
        
        return self.duplicate_partially(input_name = self.input_name , input_shape=input_shape)


# other functions
def concatenate(model1: ONNXWrap, model2: ONNXWrap) -> ONNXWrap:
    """
    Concatenates two ONNX models.

    Args:
    model1 (ONNXWrap): The first ONNX model to concatenate.
    model2 (ONNXWrap): The second ONNX model to concatenate.

    Returns:
    ONNXWrap: The concatenated ONNX model.

    Raises:
    AssertionError: If the output shape of the first model does not match the input shape of the second model, or if there is any intersection between the names in the first model and in the second model.
    """

    # Check if the output name and shape of self match the input name and shape of other
    assert model1.shape[model1.output_name] == model2.shape[model2.input_name], "Output shape of the first model and input shape of the second model should match"

    # rename any overlapping node names, input names, and output names in the second model
    names_in_model1 = model1.variables + [node.name for node in model1.nodes if node.name] + [
        initializer.name for initializer in model1.initializers]
    names_in_model2 = model2.variables + [node.name for node in model2.nodes if node.name] + [
        initializer.name for initializer in model2.initializers]
    assert len(set(names_in_model1).intersection(names_in_model2)
               ) == 0, "There must be no intersection between the names in the first model and in the second model."
    part_output_input_name = model1.output_name+model2.input_name

    part_node1 = duplicate_proto(model1.nodes[-1])
    part_node1.ClearField('output')
    part_node1.output.extend([part_output_input_name])

    part_node2 = duplicate_proto(model2.nodes[0])
    input_idx = list(part_node2.input).index(model2.input_name)
    part_node2.input.pop(input_idx)
    part_node2.input.insert(input_idx, part_output_input_name)

    # concatenate the nodes, inputs, outputs, and initializers from both models
    new_nodes = model1.nodes[:-1] + [part_node1, part_node2] + model2.nodes[1:]
    new_input_name = model1.input_name
    new_input_shape = model1.shape[model1.input_name]
    new_output_name = model2.output_name
    new_output_shape = model2.shape[model2.output_name]
    new_initializers = model1.initializers + model2.initializers

    new_wrap = model1.duplicate_partially(nodes=new_nodes, input_name=new_input_name, input_shape=new_input_shape,
                                          output_name=new_output_name, output_shape=new_output_shape, initializers=new_initializers)
    return new_wrap

    