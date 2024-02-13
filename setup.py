from setuptools import setup

requirements = [
    "numpy",
    "scanpy",
    "numba",
    "pandas",
    "scikit-learn",
    "scipy",
    "seaborn",
    "matplotlib",
    "pytest",
    "torch"
]

setup(
    name="Annotatability",
    version="0.0.5",
    packages=["Annotatability"],
    url="https://github.com/nitzanlab/Annotatability",
    license="MIT License",
    author="jonathankarin",
    author_email="jonathan.karin@mail.huji.ac.il",
    description="Interpreting single cell data using annotation-trainability analysis",
    python_requires=">=3",
    install_requires=requirements
)