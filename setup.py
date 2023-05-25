from setuptools import setup, find_packages

setup(
    name='delbugv',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'onnx',
        'onnxruntime'
    ],
    author='RayaE',
    author_email='rayae@cs.huji.ac.il',
    description='DelBugV: Delta-Debugging Neural Network Verifiers',
    url='https://github.com/Raya5/DelBugV',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
