

import numpy as np

import os
import uuid
import pickle
from datetime import datetime
import shutil
import re

from delbugve import VerificationQuery, VerificationQueryAnswer, ONNXWrap, VNNLIBWrap, VerificationStatus


dir_path = os.path.dirname(os.path.realpath(__file__))
CACHE_PATH = os.path.join(dir_path, 'cache')


def hash_query(verifier_id: str, verification_query: VerificationQuery):
    category = verification_query.category
    onnx_model = verification_query.onnx_wrap.model
    vnnlib = verification_query.vnnlib_wrap.vnnlib
    return str(uuid.uuid5(uuid.NAMESPACE_OID, '\n'.join([verifier_id, category, onnx_model.__str__(), vnnlib])))


def get_query_id(verifier_id: str, verification_query: VerificationQuery):
    query_hash = hash_query(verifier_id, verification_query)
    return query_hash


def get_query_file_path(verifier_id: str, verification_query: VerificationQuery):
    file_id = get_query_id(verifier_id, verification_query) + ".pkl"
    file_path = os.path.join(CACHE_PATH, *file_id.split('-'))
    return file_path

def get_query_testing_directory_path(verifier_id: str, verification_query: VerificationQuery):
    file_id = get_query_id(verifier_id, verification_query)
    dicetory_path = os.path.join(CACHE_PATH,'running', file_id)
    return dicetory_path

# "running time, SAT, Satisfying assignment"
# "running time, UNSAT, None"
# "running time, ERROR, error_counter"
# "running time, TIMEOUT, 0"


def get_raw_query(verifier_id: str, verification_query: VerificationQuery):
    query_file_path = get_query_file_path(verifier_id,verification_query)
    if not os.path.exists(query_file_path):
        return None
    try:
        with open(query_file_path, 'rb') as fp:
            query = pickle.load(fp)
        return query
    except:
        os.remove(query_file_path)
        return None


def get_query(verifier_path: str, verification_query: VerificationQuery) -> VerificationQueryAnswer:
    query = get_raw_query(verifier_path, verification_query)
    if query is None:
        return VerificationQueryAnswer(verification_query, verifier_path, VerificationStatus.YETTORUN)
    elif query[0] == "SAT":
        assignment = np.array(query[1], dtype=np.float32)
        running_time = query[2]
        return VerificationQueryAnswer(verification_query, verifier_path, VerificationStatus.SAT, assignment, running_time=running_time)
    elif query[0] == "UNSAT":
        running_time = query[2]
        return VerificationQueryAnswer(verification_query, verifier_path,VerificationStatus.UNSAT, running_time=running_time)
    elif query[0] == "ERROR":
        running_time = query[2]
        return VerificationQueryAnswer(verification_query, verifier_path,VerificationStatus.ERROR, running_time=running_time)
    elif query[0] == "TIMEOUT":
        running_time = query[2]
        return VerificationQueryAnswer(verification_query, verifier_path,VerificationStatus.TIMEOUT, running_time=running_time)
    else:
        raise ValueError("Invalid status: "+str(query[0]))


def save_query(verifier_id, verification_query_answer: VerificationQueryAnswer, force_override=False):
    query = get_raw_query(verifier_id, verification_query_answer.verification_query)
    assert verification_query_answer.satisfiable != VerificationStatus.YETTORUN
    save = False
    if force_override:
        data = verification_query_answer.satisfying_assignment
        if verification_query_answer.satisfiable == VerificationStatus.ERROR:
            data = 1
        save = True
    elif verification_query_answer.satisfiable in [VerificationStatus.SAT, VerificationStatus.UNSAT]:
        data = verification_query_answer.satisfying_assignment
        save = True
    elif query is None:
        data = 0
        save = True
    elif verification_query_answer.satisfiable == VerificationStatus.ERROR:
        if query[0] == "ERROR":
            data = query[1]+1
            if data > 10:
                raise RuntimeError(f"This query ({get_query_file_path(verifier_id, verification_query_answer.verification_query)}) is repeatedly triggering error. Is the query valid? Is the verifier installed correctly?")
            save = True
    elif verification_query_answer.satisfiable == VerificationStatus.TIMEOUT and query[0] == "TIMEOUT":
        if verification_query_answer.running_time > query[2]:
            data = None
            save = True

    elif verification_query_answer.satisfiable == VerificationStatus.TIMEOUT and query[0] == "ERROR":
        data = None
        save = True
    else:
        raise ValueError("Invalid status: " +
                         str(query[0]), type(verification_query_answer.satisfiable))

    if save:
        query_file_path = get_query_file_path(verifier_id, verification_query_answer.verification_query)
        os.makedirs(os.path.dirname(query_file_path), exist_ok=True)
        with open(query_file_path, 'wb') as f:
            assert verification_query_answer.running_time >= 0, f"Save the query with {verification_query_answer.running_time} running time?, verifier answer is {verification_query_answer.satisfiable}"
            pickle.dump((verification_query_answer.satisfiable.name, data,
                        verification_query_answer.running_time), f)
        with open(query_file_path[0:-3]+"ans", 'w') as f:
            f.write(verification_query_answer.verifier_path + " : " +  verification_query_answer.satisfiable.name)
            if verification_query_answer.satisfiable == VerificationStatus.SAT:
                f.write("Satisfying assignment suggested : " +  str(verification_query_answer.satisfying_assignment.flatten().tolist()))
                

def run_verifier(verifier_id: str, vnncomp_scripts_path: str, verification_query: VerificationQuery, force_override=False)-> VerificationQueryAnswer:
    category = verification_query.category
    onnx_wrap = verification_query.onnx_wrap
    vnnlib_wrap = verification_query.vnnlib_wrap
    timeout = verification_query.timeout
    query_answer = get_query(verifier_id, verification_query)
    if query_answer is None or (query_answer.satisfiable == VerificationStatus.TIMEOUT and timeout > query_answer.running_time) or query_answer.satisfiable == VerificationStatus.ERROR or query_answer.satisfiable == VerificationStatus.YETTORUN:
        testing_directory = get_query_testing_directory_path(verifier_id, verification_query)
        os.makedirs(os.path.dirname(testing_directory), exist_ok=True)
        onnx_path = os.path.join(testing_directory, "model.onnx")
        vnnlib_path = os.path.join(testing_directory, "property.vnnlib")
        output_path = os.path.join(testing_directory, "output.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx_wrap.save(onnx_path)
        vnnlib_wrap.save(vnnlib_path)
        
        prepare_script = os.path.join(
            vnncomp_scripts_path, "prepare_instance.sh")
        run_script = os.path.join(
            vnncomp_scripts_path, "run_instance.sh")
        before = datetime.now()  
        
        os.system("timeout 10m bash "
                  + prepare_script
                  + " v1 "
                  + category
                  + " "
                  + onnx_path+" "
                  + vnnlib_path)
        exit_code = os.system("timeout "
                  +str(timeout+20)
                  +"m bash "
                  + run_script
                  + " v1 "
                  + category + " "
                  + onnx_path +" "
                  + vnnlib_path+" "
                  + output_path+" "
                  + str(timeout))
        after = datetime.now()  
        running_time=(after-before).seconds
        if exit_code >> 8 == 124:
            res_vq =  VerificationQueryAnswer(verification_query, verifier_id, VerificationStatus.TIMEOUT, running_time=running_time)
        elif not os.path.exists(output_path):
            res_vq = VerificationQueryAnswer(verification_query, verifier_id, VerificationStatus.ERROR, running_time=running_time)
        else:
            with open(output_path, 'r') as f:
                res = f.readlines()
            shutil.rmtree(testing_directory)
            if res[0].strip() == "violated" and len(res) >1 :
                assignment = []
                for i in range(1, len(res)):
                    a = re.split(' |\(|\)|\n', res[i])
                    a = [x for x in a if len(x) >0]
                    if not a[0].startswith('X_'):
                        break
                    if a[0].split('_')[1] != str(i-1):
                        raise ValueError('invalid sat assignment in output file.')

                    assignment.append(float(a[1]))
                res_vq =  VerificationQueryAnswer(verification_query, verifier_id, VerificationStatus.SAT, np.array(assignment).astype(np.float32), running_time=running_time)
            elif res[0].strip() == "holds":
                res_vq =  VerificationQueryAnswer(verification_query, verifier_id, VerificationStatus.UNSAT, running_time=running_time)
            else: 
                res_vq =  VerificationQueryAnswer(verification_query, verifier_id, VerificationStatus.ERROR, running_time=running_time)
        
        save_query(verifier_id, res_vq)
        return res_vq

    else:
        return query_answer

