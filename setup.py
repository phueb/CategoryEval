from setuptools import setup

from categoryeval import __name__, __version__

setup(
    name=__name__,
    version=__version__,
    packages=[__name__],
    include_package_data=True,
    install_requires=[
        'bayesian-optimization==0.6',
        'cytoolz==0.10.1',
        'scipy==1.4.1',
        'scikit-learn==0.21.3',
        'matplotlib==3.1.2',
        'numpy==1.22.0',
        'pyitlib==0.2.2',
    ],
    url='https://github.com/phueb/CategoryEval',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Evaluate word representations for category knowledge'
)