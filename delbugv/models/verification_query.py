from __future__ import annotations

import numpy as np

from typing import Optional, Tuple, Union, Callable

from .onnx_wrap import ONNXWrap
from .vnnlib_wrap import VNNLIBWrap
from ..helper import VerificationStatus

class VerificationQuery:

    def __init__(self, onnx_wrap: ONNXWrap, vnnlib_wrap: VNNLIBWrap, category: str, timeout: Optional[int] = 900) -> None:
        """
        Represents a verification query
        """
        self.onnx_wrap: ONNXWrap = onnx_wrap
        self.vnnlib_wrap: VNNLIBWrap = vnnlib_wrap
        self.category: Optional[str] = category
        self.timeout: int = timeout
        self.satisfiable: VerificationStatus = VerificationStatus.YETTORUN
        self.satisfying_assignment: np.array = None
        self.running_time = -1

    def copy(self, ignore_assignment = False) -> VerificationQuery:
        copied_vq = VerificationQuery(self.onnx_wrap.copy(), self.vnnlib_wrap.copy(), category=self.category, timeout = self.timeout)
        if not ignore_assignment:
            copied_vq = copied_vq.assign(self.satisfiable, self.satisfying_assignment, self.running_time)
        return copied_vq
    
    def merge_inputs(self, index1, index2):
        self.vnnlib_wrap = self.vnnlib_wrap.merge_inputs(index1, index2)

    def merge_outputs(self, index1, index2):
        self.vnnlib_wrap = self.vnnlib_wrap.merge_outputs(index1, index2)
    
    def check_if_satisfies(self, input_value: np.array):
        if not self.vnnlib_wrap.satisfy_input_constraints(input_value):
            return False
        output_value = self.onnx_wrap.run(input_value)
        if not self.vnnlib_wrap.satisfy_output_constraints(output_value):
            return False
        return True


    def assign(self, satisfiable: VerificationStatus, satisfying_assignment: Optional[np.array] = None, running_time = -1) -> VerificationQuery:
        vq = VerificationQuery(self.onnx_wrap, self.vnnlib_wrap, self.category, timeout=self.timeout)

        vq.satisfiable = satisfiable
        vq.running_time = running_time
        if satisfiable == VerificationStatus.SAT:
            if satisfying_assignment is None:
                raise ValueError(
                    "Verifier claims SAT, but does't suggest a satisfying assignment")
            vq.satisfying_assignment = satisfying_assignment
        return vq

    def count_neurons(self):
        return self.onnx_wrap.count_neurons()
        
    
    def simplify(self, func: str, params, assignment):
        return getattr(self, func)(params, assignment)

    def linearize(self, params, assignment: np.ndarray):
        new_onnx = self.onnx_wrap.linearize(*params, assignment)
        copied_vnnlib = self.vnnlib_wrap.copy()
        return VerificationQuery(new_onnx, copied_vnnlib, self.category, timeout = self.timeout)


    def delete_flatten(self, params, assignment: np.ndarray):
        new_onnx = self.onnx_wrap.delete_flatten(assignment)
        copied_vnnlib = self.vnnlib_wrap.copy()
        return VerificationQuery(new_onnx, copied_vnnlib, self.category, timeout = self.timeout)

    def flatten_input(self, params, assignment: np.ndarray):
        new_onnx = self.onnx_wrap.flatten_input(assignment)
        copied_vnnlib = self.vnnlib_wrap.copy()
        return VerificationQuery(new_onnx, copied_vnnlib, self.category, timeout = self.timeout)

    
    def combine(self, params, assignment: np.ndarray) -> VerificationQuery:
        """
        builds a new instance of VQuery with two neurons in the node linearNodeName combined to one using the satisfying_assignment.
        """        
        print("combine params:", params)
        new_onnx = self.onnx_wrap.combine(*params, assignment)
        if params[0][0] == 0:
            new_vnnlib = self.vnnlib_wrap.merge_inputs(*params[1])
        else:
            new_vnnlib = self.vnnlib_wrap.copy()
        return VerificationQuery(new_onnx, new_vnnlib, self.category, timeout = self.timeout)
