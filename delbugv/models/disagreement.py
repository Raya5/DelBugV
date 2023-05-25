import os
import time
import signal

from .verification_query import VerificationQuery
from .onnx_wrap import ONNXWrap
from .vnnlib_wrap import VNNLIBWrap
from .prioritizing_strategy import PrioritizingStrategy
from ..verify import run_verifier
from ..helper import VerificationStatus

class Disagreement:
    """
    Represents a disagreement between two verifiers.
    """
    def __init__(self, verification_query: VerificationQuery, faulty_verifier_path, oracle_verifier_path, prioritizing_strategy=None):
        self._verification_query: VerificationQuery = verification_query
        self._faulty_verifier_path = faulty_verifier_path
        self._oracle_verifier_path = oracle_verifier_path
        if prioritizing_strategy is None:
            prioritizing_strategy = PrioritizingStrategy()
        self._prioritizing_strategy = prioritizing_strategy

    @property
    def verification_query(self):
        return self._verification_query

    @property
    def faulty_verifier_path(self):
        return self._faulty_verifier_path

    @property
    def faulty_verifier_answer(self):
        if not hasattr(self, '_faulty_verifier_answer'):
            self._faulty_verifier_answer = run_verifier(
                self.faulty_verifier_path, self.faulty_verifier_path, self.verification_query)
        return self._faulty_verifier_answer

    @property
    def oracle_verifier_path(self):
        return self._oracle_verifier_path

    @property
    def oracle_verifier_answer(self):
        if self.oracle_verifier_path is None:
            return self.verification_query.assign(VerificationStatus.UNAVAILABLE)
            
        if not hasattr(self, '_oracle_verifier_answer'):
            self._oracle_verifier_answer = run_verifier(
                self.oracle_verifier_path, self.oracle_verifier_path, self.verification_query)
        return self._oracle_verifier_answer
    
    @property
    def sat_verifier_assignment(self):
        if self.faulty_verifier_answer.satisfiable == VerificationStatus.SAT:
            return self.faulty_verifier_answer.satisfying_assignment
        if self.oracle_verifier_answer.satisfiable == VerificationStatus.SAT:
            return self.oracle_verifier_answer.satisfying_assignment
        raise ValueError("None of the verifiers returned SAT")
    
    @property
    def prioritizing_strategy(self):
        return self._prioritizing_strategy

    @property
    def true_disagreement(self):
        if self.faulty_verifier_answer.satisfiable == VerificationStatus.SAT and not self.faulty_verifier_answer.correct_sat():
            return True
        if self.faulty_verifier_answer.satisfiable == VerificationStatus.UNSAT and self.oracle_verifier_answer.satisfiable == VerificationStatus.SAT:
            return True
        return False

    @classmethod
    def from_paths(cls, onnx_path, vnnlib_path,category, faulty_verifier_path, oracle_verifier_path, prioritizing_strategy=None):
        onnx = ONNXWrap.from_file(onnx_path)
        vnnlib = VNNLIBWrap.from_file(vnnlib_path)
        vq = VerificationQuery(onnx, vnnlib, category)
        return cls(vq, faulty_verifier_path, oracle_verifier_path, prioritizing_strategy)

    def simplify(self, assignment, timeout=None):
        start_time = time.time()
        elapsed_time = 0
        steps = self.prioritizing_strategy.get_steps(self.verification_query, assignment)
        for i in range(len(steps)):
            # print(f"attempting step {i}: {steps[i]}")
            simplified_verification_query = self.verification_query.simplify(
                *steps[i], assignment)
            new_disagreement = Disagreement(
                simplified_verification_query, self.faulty_verifier_path, self.oracle_verifier_path)
            if new_disagreement.true_disagreement:
                # if new_disagreement.faulty_verifier_answer.satisfiable == VerificationStatus.SAT: print("faulty assignment correct:",new_disagreement.faulty_verifier_answer.correct_sat() )
                # if new_disagreement.oracle_verifier_answer.satisfiable == VerificationStatus.SAT: print("oracle assignment correct:",new_disagreement.oracle_verifier_answer.correct_sat() )
                return new_disagreement, True
            elapsed_time = time.time() - start_time

        return None, False

    def save(self, path, comments=""):
        onnx_path = os.path.join(path, "model.onnx")
        vnnlib_path = os.path.join(path, "property.vnnlib")
        output_path = os.path.join(path, "info.txt")
        self.verification_query.onnx_wrap.save(onnx_path)
        self.verification_query.vnnlib_wrap.save(vnnlib_path)
        s = ""
        s += "Faulty verifier: " + self.faulty_verifier_answer.satisfiable.name 
        if not self.faulty_verifier_answer.satisfying_assignment is None:
            s +=" ("+str(self.faulty_verifier_answer.satisfying_assignment.tolist())+")\n"
            if self.faulty_verifier_answer.correct_sat:
                s +="This assignment is correct\n"
            else:
                s +="This assignment is incorrect\n"
        s += "Oracle verifier: " + self.oracle_verifier_answer.satisfiable.name 
        if not self.oracle_verifier_answer.satisfying_assignment is None:
            s +=" ("+str(self.oracle_verifier_answer.satisfying_assignment.tolist())+")\n"
            if self.oracle_verifier_answer.correct_sat:
                s +="This assignment is correct\n"
            else:
                s +="This assignment is incorrect\n"
        s += "ONNX in " + onnx_path + "\n"
        s += "VNNLIB in " + vnnlib_path + "\n"
        s += comments + "\n"

        with open(output_path, 'w') as f:
            f.write(s)


def delta_debug(disagreement: Disagreement, timeout=86400, save_every_step="") -> Disagreement:  # 172800

    start_time = time.time()
    elapsed_time = 0
    assert disagreement.faulty_verifier_answer.satisfiable == VerificationStatus.SAT or disagreement.oracle_verifier_answer.satisfiable == VerificationStatus.SAT, f"Neither of the verifiers returned SAT. Faulty verifier returned {disagreement.faulty_verifier_answer.satisfiable.name},  Oracle verifier returned {disagreement.oracle_verifier_answer.satisfiable.name}"
    assert disagreement.true_disagreement
    if disagreement.faulty_verifier_answer.satisfiable == VerificationStatus.SAT:
        assert not disagreement.faulty_verifier_answer.correct_sat()
    satisfying_assignment = disagreement.faulty_verifier_answer.satisfying_assignment if disagreement.faulty_verifier_answer.satisfiable == VerificationStatus.SAT else disagreement.oracle_verifier_answer.satisfying_assignment
    original_disagreement = disagreement
    while elapsed_time < (timeout - 120):

        def timeout_handler(signum, frame):
            raise TimeoutError("Function execution timed out")

        simplifying_timeout = timeout - elapsed_time - 120
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(simplifying_timeout))

        try:
            reduced_disagreement, success = disagreement.simplify(disagreement.sat_verifier_assignment, 
                timeout=simplifying_timeout)
        except TimeoutError:
            success = False
            # print("Function execution timed out!")
        finally:
            signal.alarm(0)

        if success:
            disagreement = reduced_disagreement
            assert reduced_disagreement.true_disagreement
            if len(save_every_step) > 0:
                o_c = original_disagreement.verification_query.count_neurons()
                r_c = disagreement.verification_query.count_neurons()
                s = f"Original network neurons count: {o_c}\n"
                s += f"New network neurons count: {r_c}\n"
                s += f"Neurons reduced: {o_c - r_c}\n"
                print(str(o_c - r_c) + " neurons reduced so far!")
                disagreement.save(save_every_step, comments=s)
        else:
            break

        elapsed_time = time.time() - start_time
    if elapsed_time < (timeout - 120):
        # stoped before timeout
        pass
    else:
        # reached timeout
        pass
    return disagreement
