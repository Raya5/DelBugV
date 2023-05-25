from __future__ import annotations

import numpy as np

from typing import List, Dict, Tuple, Sequence
import re
import os

from ..helper import replace_all, get_parentheses_substrings


class VNNLIBWrap():
    "A wrapper for VNN-LIB properties"
    def __init__(self, vnnlib: str):
        self.inputs = 0
        self.outputs = 0
        self.input_upper_bounds = []
        self.input_lower_bounds = []
        self.output_inequality_constraints = []
        self.output_constraints_operator = "and"
        self.output_constraints_operator_update = False

        self.vnnlib = vnnlib
        substrings = get_parentheses_substrings(vnnlib)
        for substring in substrings:
            self.activate_constraints(substring)

    def copy(self) -> VNNLIBWrap:
        return VNNLIBWrap(self.vnnlib)

    @classmethod
    def from_file(cls, path: str) -> VNNLIBWrap:
        with open(path, 'r') as f:
            vnnlib: str = f.read()
        return cls(vnnlib)

    def activate_constraints(self, constraint: str):
        if constraint.startswith("(declare-const X_") and constraint.endswith(
                " Real)"):
            number_str = constraint[17:-6]
            number = int(number_str)
            assert self.inputs == number
            self.inputs += 1
            self.input_upper_bounds.append(None)
            self.input_lower_bounds.append(None)

        elif constraint.startswith("(declare-const Y_") and constraint.endswith(
                " Real)"):
            number_str = constraint[17:-6]
            number = int(number_str)
            assert self.outputs == number
            self.outputs += 1

        elif constraint.startswith("(assert (<= X_") and constraint.endswith("))"):
            mid_str = constraint[14:-2]
            mid_str_split = mid_str.split(' ')
            assert len(mid_str_split) == 2
            input_number = int(mid_str_split[0])
            upper_bound = float(mid_str_split[1])
            self.input_upper_bounds[input_number] = upper_bound

        elif constraint.startswith("(assert (>= X_") and constraint.endswith("))"):
            mid_str = constraint[14:-2]
            mid_str_split = mid_str.split(' ')
            assert len(mid_str_split) == 2
            input_number = int(mid_str_split[0])
            lower_bound = float(mid_str_split[1])
            self.input_lower_bounds[input_number] = lower_bound

        elif constraint.startswith("(and (<= Y_") and constraint.endswith(""):
            self.output_constraints_operator_update = True
            mid_str = constraint[9:-2]
            mid_str_split = mid_str.split(' ')
            assert len(mid_str_split) == 2
            y1 = int(mid_str_split[0][2:])
            y2 = int(mid_str_split[1][2:])
            self.output_inequality_constraints.append((y1, y2))

        elif constraint.startswith("(and (>= Y_") and constraint.endswith(""):
            self.output_constraints_operator_update = True
            mid_str = constraint[9:-2]
            mid_str_split = mid_str.split(' ')
            assert len(mid_str_split) == 2
            y1 = int(mid_str_split[0][2:])
            y2 = int(mid_str_split[1][2:])
            self.output_inequality_constraints.append((y2, y1))

        elif constraint.startswith("(assert (>= Y_") and constraint.endswith("))"):
            self.output_constraints_operator_update = True
            mid_str = constraint[12:-2]
            mid_str_split = mid_str.split(' ')
            assert len(mid_str_split) == 2
            y1 = int(mid_str_split[0][2:])
            y2 = int(mid_str_split[1][2:])
            self.output_inequality_constraints.append((y2, y1))

        elif constraint.startswith("(assert (<= Y_") and constraint.endswith("))"):
            self.output_constraints_operator_update = True
            mid_str = constraint[12:-2]
            mid_str_split = mid_str.split(' ')
            assert len(mid_str_split) == 2
            y1 = int(mid_str_split[0][2:])
            y2 = int(mid_str_split[1][2:])
            self.output_inequality_constraints.append((y1, y2))

        elif constraint.startswith("(assert (or") and constraint.endswith("))"):
            assert 'X' not in constraint
            substrings = get_parentheses_substrings(constraint[11:-2])
            assert not self.output_constraints_operator_update
            self.output_constraints_operator = 'or'
            self.output_constraints_operator_update = True
            for substring in substrings:
                self.activate_constraints(substring)
        else:
            raise ValueError("Unable to interpret the vnn-lib: " + constraint)

    @property
    def lower_bounds(self) -> np.ndarray:
        print(self.input_lower_bounds)
        return np.array(self.input_lower_bounds, dtype=np.float32)
    @property
    def upper_bounds(self) -> np.ndarray:
        return np.array(self.input_upper_bounds, dtype=np.float32)
    def random_input(self) -> np.ndarray:
        r = np.random.uniform(size=self.lower_bounds.shape)
        r = (r * [self.upper_bounds - self.lower_bounds] + self.lower_bounds)[0]
        return r

    def random_corner_input(self) -> np.ndarray:
        r = np.random.randint(2, size=self.lower_bounds.shape)
        r = (r * [self.upper_bounds - self.lower_bounds] + self.lower_bounds)[0]
        return r

    def satisfy_input_constraints(self, input_assignment: np.ndarray):
        assignment = input_assignment.flatten()
        if assignment.shape[0] != self.inputs:
            raise ValueError(f"input_assignment shape: {input_assignment.shape} doesn't match the number of inputs in vnnlib file ({self.inputs} inputs).")

        for i in range(self.inputs):
            lb = self.input_lower_bounds[i]
            ub = self.input_upper_bounds[i]
            x = assignment[i]
            if x < lb or x > ub:
                # print(assignment.shape)
                # print(i, ':', lb, '<', x, '<', ub)
                return False
        return True

    def satisfy_output_constraints(self, output_assignment: np.ndarray):
        assignment = output_assignment.flatten()
        if assignment.shape[0] != self.outputs:
            raise ValueError(f"output_assignment shape: {output_assignment.shape} doesn't match the number of outputs in vnnlib file ({self.outputs} outputs).")
        satisfied_constraints = 0 
        for output_constraint in self.output_inequality_constraints:
            if assignment[output_constraint[0]] <= assignment[output_constraint[1]]:
                satisfied_constraints +=1
        if self.output_constraints_operator == 'or' and  satisfied_constraints:
            return True
        elif self.output_constraints_operator == 'and' and  satisfied_constraints == len(self.output_inequality_constraints):
            return True
        else:
            return False

    def merge_inputs(self, index1: int, index2: int) -> VNNLIBWrap:
        index1, index2 = min(index1, index2), max(index1, index2)
        txt = ''
        index1_txt = ''
        index2_txt = ''
        text = self.vnnlib.split("\n")
        inputs_num = index2
        for line in text:
            if line.strip()[:1] == ';':
                txt += line + "\n"
                continue
            if not ' X_' + str(index2) + ' ' in line:
                txt += line + "\n"
            else:
                index2_txt += line
            if ' X_' + str(index1) + ' ' in line:
                index1_txt += line

            if ' X_' in line:
                idx = int(line.split('_')[1].split(' ')[0])
                inputs_num = max(inputs_num, idx)

        # calculating lower bound
        x1_l = re.search(
            "\s*\(assert\s+\(\s*>=\s*X_([0-9]+)\s+(-?[0-9.]+)\s*\)\s*\)", index1_txt)
        x1_l_num = int(x1_l.group(1))
        if x1_l_num != index1:
            raise ValueError("Unexpected structure in vnnlib file")
        x1_lb = float(x1_l.group(2))
        x2_l = re.search(
            "\s*\(assert\s+\(\s*>=\s*X_([0-9]+)\s+(-?[0-9.]+)\s*\)\s*\)", index2_txt)
        x2_l_num = int(x2_l.group(1))
        if x2_l_num != index2:
            raise ValueError("Unexpected structure in vnnlib file")
        x2_lb = float(x2_l.group(2))
        lb = (x1_lb + x2_lb) / 2

        # calculating upper bound
        x1_u = re.search(
            "\s*\(assert\s+\(\s*<=\s*X_([0-9]+)\s+(-?[0-9.]+)\s*\)\s*\)", index1_txt)
        x1_u_num = int(x1_u.group(1))
        if x1_u_num != index1:
            raise ValueError("Unexpected structure in vnnlib file")
        x1_ub = float(x1_u.group(2))
        x2_u = re.search(
            "\s*\(assert\s+\(\s*<=\s*X_([0-9]+)\s+(-?[0-9.]+)\s*\)\s*\)", index2_txt)
        x2_u_num = int(x2_u.group(1))
        if x2_u_num != index2:
            raise ValueError("Unexpected structure in vnnlib file")
        x2_ub = float(x2_u.group(2))
        ub = (x1_ub + x2_ub) / 2

        rep_dict = {
            x1_l.group(0).strip(): "(assert (>= X_" + str(x1_l_num) + " " + "{:.15f}".format(lb) + "))",
            x1_u.group(0).strip(): "(assert (<= X_" + str(x1_u_num) + " " + "{:.15f}".format(ub) + "))"

        }
        for i in range(index2, inputs_num):
            rep_dict[' X_' + str(i + 1) + ' '] = ' X_' + str(i) + ' '

        txt = replace_all(txt, rep_dict)
        return VNNLIBWrap(txt)

    def merge_outputs(self, index1: int, index2: int) -> VNNLIBWrap:
        raise NotImplementedError(
            "yet to import - need to deal with inequities with more than one Y parameter")

    def save(self, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(self.vnnlib)
