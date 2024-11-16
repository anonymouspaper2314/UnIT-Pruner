from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='rtp',
    version='0.1.1',
    author='Ashe Neth',
    author_email='aneth@wpi.edu',
    description='Runtime pruning for PyTorch',
    install_requires=[
        'numpy',
        'torch'
    ],
    packages=[
        'rtp'
    ],
    ext_modules=[
        cpp_extension.CppExtension(
            'rtp_core',
            [
                'rtp/rtp_core.cpp'
            ]
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
)

