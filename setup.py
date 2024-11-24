from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# List of dependencies
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
    packages=find_packages(),  # Automatically detect all packages
    url="https://github.com/nitzanlab/Annotatability",
    license="MIT License",
    author="jonathankarin",
    author_email="jonathan.karin@mail.huji.ac.il",
    description="Interpreting single cell data using annotation-trainability analysis",
    long_description=long_description,  # Correctly read from README.md
    long_description_content_type="text/markdown",  # Specify content type
    python_requires=">=3.7",  # Specify a specific Python version or range
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
