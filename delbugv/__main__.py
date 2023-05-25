
import sys
from pathlib import Path
from delbugv import Disagreement, DeltaDebugDisagreement, VerificationQuery, ONNXWrap, VNNLIBWrap

def main():
    if (args_count := len(sys.argv)) > 9:
        print(f"7 or 8 arguments expected, got {args_count - 1}")
        raise SystemExit(2)
    elif args_count < 8:
        print("usage: <onnx-network-path> <vnnlib-property-path> <category> <onnx-network-output-path> <vnnlib-property-output-path> <timeout> <faulty-verifier-vnncomp-scripts-path> <optional: oracle-verifier-vnncomp-scripts-path>")
        raise SystemExit(2)
    
    onnx_path = Path(sys.argv[1])
    vnnlib_path = Path(sys.argv[2])
    category = Path(sys.argv[3])
    onnx_output_path = Path(sys.argv[4])
    vnnlib_output_path = Path(sys.argv[5])
    timeout = Path(sys.argv[6])
    faulty_verifier_path = Path(sys.argv[7])
    if args_count > 8:
        oracle_verifier_path = Path(sys.argv[8])
    else:
        oracle_verifier_path= None
        
    onnx_wrap = ONNXWrap.from_file(onnx_path)
    vnnlib_wrap = VNNLIBWrap.from_file(vnnlib_path)
    verification_query = VerificationQuery(onnx_wrap, vnnlib_wrap, category)
    disagreement = Disagreement(verification_query, faulty_verifier_path, oracle_verifier_path)
    
    reduced_disagreement = DeltaDebugDisagreement(disagreement)
    
    reduced_disagreement.verification_query.onnx_wrap.save(onnx_output_path)
    reduced_disagreement.verification_query.vnnlib_wrap.save(vnnlib_output_path)

    
if __name__ == '__main__':
    main()