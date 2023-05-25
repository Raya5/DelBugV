# DelBugV


DelbugV is an artifact that ues delta-debugging techniques to automatically reduce bug-triggering inputs of neural
networks verifiers. 

## Background

In the International Verification of Neural Networks Competitions [VNN-COMP21](https://github.com/stanleybak/vnncomp2021) and  [VNN-COMP22](https://github.com/stanleybak/vnncomp2022), 
verifiers disagreed on certain verification queries and returned different answers. 
Incorrect answer are the result of implementation bugs in the verifiers. 
With DelbugV we hope to help find those faster. 


## Usage

DelbugV was designed to match the standard formats used in the VNN-COMP. 
The format of the neural networks and the properties is the VNN-LIB format.
In launching the verifiers, we used the format suggested int thec as well. 
This scripts needed are explained [here](https://github.com/stanleybak/vnncomp2021/blob/main/README.md#scripts).
We suggest using the verifiers from the VNN-COMP for testing as they already adhere to the requirements of the VNN-LIB to launch.

To initiate the delta-debuging process, run:
```
python delbugv <onnx-network-path> <vnnlib-property-path> <category> <onnx-network-output-path> <vnnlib-property-output-path> <timeout> <faulty-verifier-vnncomp-scripts-path> <optional: oracle-verifier-vnncomp-scripts-path>
```


The parameters are:

- onnx-network-path: is the path to the onnx model file

- vnnlib-property-path: is the path to the vnnlib file

- category: the category of the verification query (matches the category from VNN-COMP)

- onnx-network-output-path: is the path to the file in which to save the simplified onnx model

- vnnlib-property-output-path: is the path to the file in which to save the simplified vnnlib property

- timeout: the process will returned the most simplified query it calculated by this timeout

- faulty-verifier-vnncomp-scripts-path: is the path to the vnncomp_scripts folder of the faulty verifier

- oracle-verifier-vnncomp-scripts-path: is the path to the vnncomp_scripts folder of the oracle verifier (mandatory if the faulty verifier returns UNSAT answer on the query)




