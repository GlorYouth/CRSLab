from setuptools import setup, find_packages

try:
    import torch
    import torch_geometric
except Exception:
    raise Exception('Please install PyTorch and PyTorch Geometric manually first.\n' +
                    'View CRSLab GitHub page for more information: https://github.com/RUCAIBox/CRSLab')

setup_requires = []

install_requires = [
    'loguru<=0.7.3',
    'tqdm<=4.67.1',
    'numpy<=2.2.5',
    'torch-geometric<=2.6.1',
    'transformers<=4.51.3',
    'pyyaml<=6.0.2',
    'nltk<=3.9.1',
    'fuzzywuzzy<=0.18.0',
    'fasttext<=0.9.3',
    'scikit-learn<=1.6.1',
    'requests<=2.32.3',
    'setuptools<80.0.0'
]

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Human Machine Interfaces"
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='crslab',
    version='0.1.2',  # please remember to edit crslab/__init__.py in response, once updating the version
    author='CRSLabTeam',
    author_email='francis_kun_zhou@163.com',
    description='An Open-Source Toolkit for Building Conversational Recommender System(CRS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RUCAIBox/CRSLab',
    packages=[
        package for package in find_packages()
        if package.startswith('crslab')
    ],
    classifiers=classifiers,
    install_requires=install_requires,
    setup_requires=setup_requires,
    python_requires='>=3.6',
)
