from setuptools import find_packages, setup

setup(
    name="banking-ml-framework",
    version="0.1.0",
    description="Reusable banking machine learning framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
