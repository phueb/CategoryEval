from setuptools import setup

from categoryeval import __name__, __version__

setup(
    name=__name__,
    version=__version__,
    packages=[__name__],
    include_package_data=True,
    install_requires=[
        'bayesian-optimization==0.6',
        'cached_property',
        'sortedcontainers',
    ],
    url='https://github.com/phueb/CategoryEval',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Evaluate category knowledge from word representations'
)