import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scipy_cut_tree_balanced",
    version="1.0.1",
    author="Vicente Reyes-Puerta",
    author_email="vr.github@hotmail.com",
    description="Python function that performs a balanced cut tree of a SciPy linkage matrix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vreyespue/scipy_cut_tree_balanced",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)