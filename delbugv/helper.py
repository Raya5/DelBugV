import onnxruntime as ort
import numpy as np
import onnx

from typing import Any, Tuple
import enum


class VerificationStatus(enum.Enum):
    SAT = 0
    UNSAT = 1
    YETTORUN = 2
    ERROR = 3
    TIMEOUT = 4
    UNAVAILABLE = 5


def duplicate_proto(proto: onnx._Proto) -> onnx._Proto:
    """Duplicate an ONNX protobuf message.

    Args:
        proto: An ONNX protobuf message to be duplicated.
            A message of type 'onnx._Proto'.

    Returns:
        A duplicate of the input protobuf message.
        A message of type 'onnx._Proto'.
    """
    serialized_proto = onnx._serialize(proto)
    duplicated_proto = onnx._deserialize(serialized_proto, type(proto)())
    return duplicated_proto


def merge_to_linearize(weights_1: np.ndarray, biases_1: np.ndarray, activated_neurons: np.ndarray, weights_2: np.ndarray, biases_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Combines two sets of weights and biases to create a new weights and biases that replace them.

    This function linearizes a neural network by combining two sets of weights and biases 
    with an array of activated neurons of the ReLU neuron. The activated neurons are represented as a 1D array 
    and are used to replace ReLU activation functions. The combined weights and biases create 
    a new set of weights and biases for the new linearized layer.

    Args:
        weights_1: A 2D numpy array of weights for the first layer of the neural network.
                Shape (m, n).
        biases_1: A 1D numpy array of biases for the first layer of the neural network.
                Shape (n,).
        activated_neurons: A 1D numpy array of activated neurons.
                        Shape (n,).
        weights_2: A 2D numpy array of weights for the second layer of the neural network.
                Shape (n, p).
        biases_2: A 1D numpy array of biases for the second layer of the neural network.
                Shape (p,).

    Returns:
        A tuple containing two 2D numpy arrays of weights and biases for the new linearized layer.
        The first element of the tuple is a 2D numpy array of weights. Shape (m, p).
        The second element of the tuple is a 1D numpy array of biases. Shape (p,).

    Raises:
        AssertionError: If the shape of the activated_neurons array is not 1D.
    """
    assert len(activated_neurons.shape) == 1
    relu_replacement = np.diagflat(activated_neurons)
    new_wights = weights_1 @ relu_replacement @ weights_2
    new_biases = biases_1 @ relu_replacement @ weights_2 + biases_2

    return new_wights.astype(np.float32),  new_biases.astype(np.float32)


def merge_to_combine(weights_1: np.ndarray, biases_1: np.ndarray, weights_2: np.ndarray, first_neuron_idx: int, second_neuron_idx, mid_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    method = ""
    neuron_1_value = mid_output[first_neuron_idx]
    neuron_2_value = mid_output[second_neuron_idx]
    # print("neurons idx:", first_neuron_idx, second_neuron_idx)
    # print("neurons values:", neuron_1_value, neuron_2_value)
    assert np.sign(neuron_1_value) != 0 and np.sign(neuron_2_value) != 0

    if np.sign(neuron_1_value) == np.sign(neuron_2_value) == -1:  # -1,-1

        new_weights_1 = np.array(weights_1).T
        new_weights_1[first_neuron_idx] = (
            new_weights_1[first_neuron_idx]+new_weights_1[second_neuron_idx])/2
        new_weights_1 = np.delete(new_weights_1, second_neuron_idx, 0).T

        new_biases_1 = np.array(biases_1)
        new_biases_1[first_neuron_idx] = (
            new_biases_1[first_neuron_idx]+new_biases_1[second_neuron_idx])/2
        new_biases_1 = np.delete(new_biases_1, second_neuron_idx, 0)

        new_weights_2 = np.array(weights_2)
        new_weights_2[first_neuron_idx] = (
            new_weights_2[first_neuron_idx]+new_weights_2[second_neuron_idx])/2
        new_weights_2 = np.delete(new_weights_2, second_neuron_idx, 0)

    elif np.sign(neuron_1_value) == np.sign(neuron_2_value) == 1:
        after_relu = np.maximum(0, mid_output)
        assert np.all(after_relu[[first_neuron_idx, second_neuron_idx]] == mid_output[[
                      first_neuron_idx, second_neuron_idx]])
        selected_neurons = after_relu[[first_neuron_idx, second_neuron_idx]]
        selected_neurons_mean = selected_neurons.sum()/2
        selected_neurons_weights = weights_2[[
            first_neuron_idx, second_neuron_idx], :]
        after_multiplication = selected_neurons @ selected_neurons_weights

        new_weights_1 = np.array(weights_1).T
        new_weights_1[first_neuron_idx] = (
            new_weights_1[first_neuron_idx]+new_weights_1[second_neuron_idx])/2
        new_weights_1 = np.delete(new_weights_1, second_neuron_idx, 0).T
        new_biases_1 = np.array(biases_1)
        new_biases_1[first_neuron_idx] = (
            new_biases_1[first_neuron_idx]+new_biases_1[second_neuron_idx])/2
        new_biases_1 = np.delete(new_biases_1, second_neuron_idx, 0)

        new_selected_neurons = selected_neurons @ selected_neurons_weights
        selected_neurons_mean = selected_neurons.sum()/2

        new_weights_2 = np.array(weights_2)
        new_weights_2[first_neuron_idx] = after_multiplication / \
            selected_neurons_mean
        new_weights_2 = np.delete(new_weights_2, second_neuron_idx, 0)
    else:
        negative_value_idx = first_neuron_idx
        positive_value_idx = second_neuron_idx
        if np.sign(neuron_1_value) > 0:
            negative_value_idx = second_neuron_idx
            positive_value_idx = first_neuron_idx
            assert np.sign(neuron_2_value) < 0

        new_weights_1 = np.array(weights_1).T
        new_weights_1 = np.delete(new_weights_1, negative_value_idx, 0).T
        new_biases_1 = np.array(biases_1)
        new_biases_1 = np.delete(new_biases_1, negative_value_idx, 0)

        new_weights_2 = np.array(weights_2)
        new_weights_2 = np.delete(new_weights_2, negative_value_idx, 0)

    return new_weights_1.astype(np.float32),  new_biases_1.astype(np.float32), new_weights_2.astype(np.float32)



def merge_to_combine_inputs(weights_1: np.ndarray, first_neuron_idx: int, second_neuron_idx, assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    method = ""
    neuron_1_value = assignment[first_neuron_idx]
    neuron_2_value = assignment[second_neuron_idx]
    assert np.sign(neuron_1_value) != 0 and np.sign(neuron_2_value) != 0

    new_weights_1 = np.array(weights_1)
    new_weights_1[first_neuron_idx] = (
        new_weights_1[first_neuron_idx]+new_weights_1[second_neuron_idx])/2
    new_weights_1 = np.delete(new_weights_1, second_neuron_idx, 0)


    return new_weights_1.astype(np.float32)


def assertions_gemm_node_attributes(node: onnx.NodeProto) -> None:
    """Perform assertions on the attributes of a GEMM node in an ONNX graph.

    This function performs assertions on the attributes of a GEMM node in an ONNX graph. 
    The GEMM node is identified by its 'op_type' attribute. 
    The function then checks the value of the 'alpha', 'beta', and 'transB' attributes 
    of the node, and raises an AssertionError if they do not match the expected values.

    Args:
        node: An ONNX NodeProto object representing a GEMM node.

    Raises:
        AssertionError: If the 'op_type' attribute of the node is not "Gemm", 
                        or if the values of 'alpha', 'beta', or 'transB' attributes do not match 
                        the expected values (1.0 for 'alpha' and 'beta', and 1 for 'transB').
    """
    assert node.op_type == "Gemm"
    for att in node.attribute:
        if att.name == "alpha":
            assert att.f == 1.0
        elif att.name == "beta":
            assert att.f == 1.0
        elif att.name == "transB":
            assert att.i == 1


def assertions_conv_node_attributes(node: onnx.NodeProto):
    """Perform assertions on the attributes of a Conv node in an ONNX graph.

    This function performs assertions on the attributes of a Conv node in an ONNX graph. 
    The Conv node is identified by its 'op_type' attribute. 
    The function then checks the value of the 'group', 'kernel_shape', 'pads', and 'strides' attributes 
    of the node, and raises an AssertionError if they do not match the expected values.

    Args:
        node: An ONNX NodeProto object representing a Conv node.

    Raises:
        AssertionError: If the 'op_type' attribute of the node is not "Conv", 
                        or if the values of 'group', 'kernel_shape', 'pads', or 'strides' 
                        attributes do not match the expected values (1 for 'group', [4, 4] for 
                        'kernel_shape', [1, 1, 1, 1] for 'pads', and [2, 2] for 'strides').
    """
    assert node.op_type == "Conv"
    for att in node.attribute:
        if att.name == "group":
            assert att.i == 1
        elif att.name == "kernel_shape":
            assert att.ints == [4, 4]
        elif att.name == "pads":
            assert att.ints == [1, 1, 1, 1]
        elif att.name == "strides":
            assert att.ints == [2, 2]


def conv_to_gemm(conv_node: onnx.NodeProto, conv_input_shape: np.ndarray, weights_data: np.ndarray, bias_data: np.ndarray, assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a convolutional layer in neural network layer into a GEMM layer using the provided parameters.

    Args:
    - conv_node (onnx.NodeProto): The convolutional node of the neural network layer.
    - conv_input_shape (np.ndarray): The shape of the input tensor.
    - weights_data (np.ndarray): The weights of the convolutional layer.
    - bias_data (np.ndarray): The bias of the convolutional layer.
    - assignment (np.ndarray): The assignment of the convolutional node.

    Returns:
    A tuple containing the converted weights and biases in the GEMM format.
    """
    input_name = conv_node.input[0]
    output_name = conv_node.output[0]
    weights_name = conv_node.input[1]
    bias_name = conv_node.input[2]

    # kernel_shape
    assert conv_node.attribute[2].name == "kernel_shape"
    kernel_shape = list(conv_node.attribute[2].ints)
    # last constrain can be removed
    assert len(
        kernel_shape) == 2 and kernel_shape[0] == kernel_shape[1] and kernel_shape[0] == 4
    kernel_size = kernel_shape[0]
    assert weights_data.shape[-1] == weights_data.shape[-2] == kernel_size

    # pads
    assert conv_node.attribute[3].name == "pads"
    pads = list(conv_node.attribute[3].ints)
    # last constrain can be removed
    assert len(
        pads) == 4 and pads[0] == pads[1] == pads[2] == pads[3] and pads[0] == 1
    pads_size = pads[0]

    # strides
    assert conv_node.attribute[4].name == "strides"
    strides = list(conv_node.attribute[4].ints)
    # last constrain can be removed
    assert len(strides) == 2 and strides[0] == strides[1] and strides[0] == 2
    strides_size = strides[0]

    # in_image_size
    assert conv_input_shape[-1] == conv_input_shape[-2]
    in_image_shape = conv_input_shape[-2:]
    in_image_size = conv_input_shape[-1]

    # batch_size
    assert conv_input_shape[0] == 1
    batch_size = 1

    # channels
    in_channels = conv_input_shape[1]
    assert weights_data.shape[1] == in_channels
    out_channels = weights_data.shape[0]

    # make conv node and graph
    # make input tensor
    conv_input_tensor = onnx.helper.make_tensor_value_info(
        input_name, onnx.TensorProto.FLOAT, conv_input_shape)

    # make output tensor
    out_image_size = (in_image_shape[0] + 2 *
                      pads[0] - kernel_shape[0]) // strides[0] + 1
    out_image_shape = [out_image_size] * 2
    conv_output_shape = [batch_size, out_channels, *out_image_shape]
    conv_output_tensor = onnx.helper.make_tensor_value_info(
        output_name, onnx.TensorProto.FLOAT, conv_output_shape)

    # make initializers
    conv_weight_init = onnx.numpy_helper.from_array(
        weights_data, name=weights_name)
    conv_bias_init = onnx.numpy_helper.from_array(
        bias_data, name=bias_name)

    # make the model
    conv_graph = onnx.helper.make_graph([conv_node], 'test_conv', [conv_input_tensor], [
                                        conv_output_tensor], [conv_weight_init, conv_bias_init])
    conv_model = onnx.helper.make_model(conv_graph)

    # make gemm node and graph
    # make input tensor
    gemm_input_shape = [1, int(np.prod(conv_input_shape))]
    gemm_input_tensor = onnx.helper.make_tensor_value_info(
        input_name, onnx.TensorProto.FLOAT, gemm_input_shape)

    # make output tensor
    gemm_output_shape = [1, int(np.prod(conv_output_shape))]
    gemm_output_tensor = onnx.helper.make_tensor_value_info(
        output_name, onnx.TensorProto.FLOAT, gemm_output_shape)

    # make initializers
    gemm_weights_shape = [int(np.prod(conv_input_shape)),
                          int(np.prod(conv_output_shape))]
    gemm_bias_shape = [int(np.prod(conv_output_shape))]
    gemm_weight = []
    gemm_bias = []
    in_image_padded_size = in_image_size + 2 * pads_size
    square_coordinates = []

    square_coordinates = [(i, j) for i in range(0, in_image_padded_size-kernel_size+1, strides_size)
                          for j in range(0, in_image_padded_size-kernel_size+1, strides_size)]

    assert len(square_coordinates) == np.prod(out_image_shape)
    for out_channel_idx in range(out_channels):
        for square_offset in square_coordinates:
            square_idx_start_i, square_idx_start_j = square_offset
            kernel_start_i, kernel_start_j = 0, 0
            square_idx_end_i, square_idx_end_j = square_idx_start_i + \
                kernel_size, square_idx_start_j+kernel_size
            kernel_end_i, kernel_end_j = kernel_size, kernel_size
            while square_idx_start_i < pads_size:
                square_idx_start_i += 1
                kernel_start_i += 1
            while square_idx_start_j < pads_size:
                square_idx_start_j += 1
                kernel_start_j += 1
            while square_idx_end_i > in_image_size+pads_size:
                square_idx_end_i -= 1
                kernel_end_i -= 1
            while square_idx_end_j > in_image_size+pads_size:
                square_idx_end_j -= 1
                kernel_end_j -= 1
            square_idx_start_i -= pads_size
            square_idx_end_i -= pads_size
            square_idx_start_j -= pads_size
            square_idx_end_j -= pads_size
            zero_template = np.zeros([in_channels, *in_image_shape])
            for in_channel_idx, in_channel_offset in enumerate(range(0, in_channels*int(np.prod(in_image_shape)), int(np.prod(in_image_shape)))):
                zero_template[in_channel_idx][square_idx_start_i:square_idx_end_i, square_idx_start_j:
                                              square_idx_end_j] += weights_data[out_channel_idx][in_channel_idx][kernel_start_i:kernel_end_i, kernel_start_j: kernel_end_j]
            gemm_weight.append(zero_template.flatten())
            gemm_bias.append(bias_data[out_channel_idx])

    gemm_weight_data = np.array(gemm_weight).astype(np.float32)
    assert gemm_weights_shape == list(gemm_weight_data.T.shape), str(
        gemm_weights_shape)+"?=="+str(list(gemm_weight_data.T.shape))
    gemm_bias_data = np.array(gemm_bias).astype(np.float32)
    assert gemm_bias_shape == list(gemm_bias_data.shape)

    gemm_weight_init = onnx.numpy_helper.from_array(
        gemm_weight_data, name=weights_name)
    gemm_bias_init = onnx.numpy_helper.from_array(
        gemm_bias_data, name=bias_name)

    # make the model
    gemm_node = onnx.helper.make_node('Gemm', inputs=[
                                      input_name, weights_name, bias_name], outputs=[output_name], transB=1)
    gemm_graph = onnx.helper.make_graph([gemm_node], 'test_gemm', [gemm_input_tensor], [
                                        gemm_output_tensor], [gemm_weight_init, gemm_bias_init])
    gemm_model = onnx.helper.make_model(gemm_graph)

    # test running conv model:
    conv_session = ort.InferenceSession(conv_model.SerializeToString())
    conv_output = conv_session.run(
        None, {input_name: assignment.reshape(conv_input_shape)})[0].flatten()

    # test running gemm model:
    gemm_session = ort.InferenceSession(gemm_model.SerializeToString())
    gemm_output = gemm_session.run(
        None, {input_name: assignment.reshape(gemm_input_shape)})[0].flatten()

    if not np.all(np.isclose(conv_output, gemm_output)):
        mismatches = np.size(np.isclose(conv_output, gemm_output)) - \
            np.count_nonzero(np.isclose(conv_output, gemm_output))
        # if mismatches > 5:
        #     print(
        #         f"Due to Conv node conversion to Gemm node {mismatches} were found")

    return gemm_weight_data.T, gemm_bias_data


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def get_parentheses_substrings(s: str):
    vnnlib_lines = s.splitlines(True)
    substrings = []
    open_parentheses = 0
    curr_substring = ""
    for line_num, line in enumerate(vnnlib_lines):
        if line.strip() and line.strip()[0] == ';':
            continue
        for char in line:
            curr_substring += char
            if char == '(':
                open_parentheses += 1
                if open_parentheses == 1:
                    curr_substring = char
            elif char == ')':
                open_parentheses -= 1
                if open_parentheses < 0:
                    raise ValueError(
                        f"invalid VNN-LIB file. Line {line_num+1} has an extra \')\'")
                elif open_parentheses == 0:
                    substrings.append(curr_substring)

    return substrings
