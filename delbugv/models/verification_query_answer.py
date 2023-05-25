from __future__ import annotations

import numpy as np

from typing import Optional, Tuple, Union

from .verification_query import VerificationQuery
from ..helper import VerificationStatus

class VerificationQueryAnswer:

    def __init__(self, verification_query: VerificationQuery, verifier_path: str, satisfiable: VerificationStatus, satisfying_assignment: Optional[np.ndarray] = None, running_time = -1):
        self._verification_query: VerificationQuery= verification_query
        self._verifier_path: str = verifier_path
        self._satisfiable: VerificationStatus = satisfiable
        self._satisfying_assignment: np.array = satisfying_assignment
        self.running_time = running_time


    @property
    def verification_query(self) -> VerificationQuery:
        return self._verification_query
    
    @property
    def verifier_path(self) -> str:
        return self._verifier_path
    
    @property
    def satisfiable(self) -> VerificationStatus:
        return self._satisfiable
    
    @property
    def satisfying_assignment(self) -> Optional[np.ndarray]:
        return self._satisfying_assignment

    def correct_sat(self):
        if not self.verification_query.vnnlib_wrap.satisfy_input_constraints(self.satisfying_assignment):
            return False
        output_value = self.verification_query.onnx_wrap.run(self.satisfying_assignment)
        if not self.verification_query.vnnlib_wrap.satisfy_output_constraints(output_value):
            return False
        if self.verification_query.satisfiable != VerificationStatus.SAT:
            self.verification_query.assign(VerificationStatus.SAT, self.satisfying_assignment)
        return True
