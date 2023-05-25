
import numpy as np

from .verification_query import VerificationQuery
from .onnx_wrap import ONNXWrap
class PrioritizingStrategy:

    @classmethod
    def get_steps(cls, verification_query: VerificationQuery, assignment: np.ndarray):
        methods_by_priority = [VerificationQuery.linearize, VerificationQuery.combine]
        steps = [] # method-name, parameters
        steps += cls.get_steps_to_clean_neutral_nodes(verification_query, assignment)
        steps += cls.get_steps_to_linearize(verification_query, assignment)
        steps += cls.get_steps_to_merge_neurons(verification_query, assignment)
        return steps

    @classmethod
    def get_steps_to_linearize(cls, verification_query: VerificationQuery, assignment: np.ndarray):
        onnx_wrap : ONNXWrap = verification_query.onnx_wrap
        sequences_types = onnx_wrap.get_linearizable_sequences()
        sequences_priority = [[], [], []] #conv&fc, fc&fc, conv&conv
        for seq, seq_types in sequences_types:
            priority = 0
            if seq_types == ['Add', 'Relu', 'MatMul']:
                priority = 2
            elif seq_types == ['Gemm', 'Relu', 'Gemm']:
                priority = 2
            elif seq_types == ['Conv', 'Relu', 'Flatten']:
                priority = 1
            elif seq_types == ['Conv', 'Relu', 'Conv']:
                priority = 3
            else:
                raise ValueError("Unknown sequence: " + str(seq_types))
            assert priority > 0 
            sequences_priority[priority-1].append(seq)
        sequences_priority_sorted = []
        for priority in sequences_priority:
            sequences_priority_sorted+=sorted(priority, key=lambda x: x[0], reverse=True)
        sequences_priority_sorted_with_method = [("linearize", x) for x in sequences_priority_sorted]
        return sequences_priority_sorted_with_method
                

    @classmethod
    def get_steps_to_merge_neurons(cls, verification_query: VerificationQuery, assignment: np.ndarray):
        onnx_wrap : ONNXWrap= verification_query.onnx_wrap
        neuron_pairs = onnx_wrap.get_combinable_neuron_pairs_values(assignment)
        
        sequences_priority = [[], [], []] #inactive, active, both
        for pair in neuron_pairs:
            values = neuron_pairs[pair]
            priority = 0
            if  values[0]<=0 and values[1] <= 0:
                priority = 1
            elif values[0]>0 and values[1] > 0:
                priority = 2
            elif  values[0]<=0 and values[1] >= 0:
                priority = 3
            elif  values[0]>=0 and values[1] <= 0:
                priority = 3
            else:
                raise RuntimeError("Not supposed to reach this error")
            assert priority > 0 
            sequences_priority[priority-1].append((pair, np.abs(values[0]-values[1])))
        sequences_priority_sorted = []
        for priority in sequences_priority:
            sequences_priority_sorted+=sorted(priority, key=lambda x: x[1])
        sequences_priority_sorted_with_method = [("combine", x[0]) for x in sequences_priority_sorted]
        return sequences_priority_sorted_with_method


    @classmethod
    def get_steps_to_clean_neutral_nodes(cls, verification_query: VerificationQuery, assignment: np.ndarray):
        sequences_priority = []
        first_node = verification_query.onnx_wrap.nodes[0]
        second_node = verification_query.onnx_wrap.nodes[1]
        if  first_node.op_type == "Flatten" and second_node.op_type == "Gemm":
            # sequences_priority.append(("delete_flatten", ()))
            
            input_shape = verification_query.onnx_wrap.shape[verification_query.onnx_wrap.input_name]
            if np.max(input_shape) != np.prod(input_shape):
                sequences_priority.append(("flatten_input", ()))
            
        return sequences_priority
        