import setuptools

with open("README.md", "r", encoding="utf-8") as input_file:
    long_description = ""
    for line in input_file:
        if (not line.startswith("The following figure")) and (
            not line.startswith("![Dendrogram")
        ):
            long_description += line

setuptools.setup(
    name="scipy_cut_tree_balanced",
    version="1.1",
    author="Vicente Reyes-Puerta",
    author_email="vr.github@outlook.com",
    description="Python function that performs a balanced cut tree of a SciPy linkage matrix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vreyespue/scipy_cut_tree_balanced",
    download_url="https://github.com/vreyespue/scipy_cut_tree_balanced/archive/v_1_1.tar.gz",
    packages=setuptools.find_packages(),
    install_requires=["scipy", "numpy", "pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)