from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gnnshap_cuda_extension',  # Name of the package
    ext_modules=[
        CUDAExtension(
            name='gnnshap_cuda_extension',
            sources=['cppextension/cudagnnshap.cu'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch'],
    version='1.0.1',
    description='GNNShap Cuda Extension',
)