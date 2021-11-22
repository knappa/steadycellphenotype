from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="steady_cell_phenotype",
    version="0.3.1",
    author="A. C. Knapp",
    author_email="adam.knapp@medicine.ufl.edu",
    description="A tool for the computation of steady states and exploration of dynamics "
    "in ternary intracellular biological networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knappa/steadycellphenotype",
    project_urls={
        "Bug Tracker": "https://github.com/knappa/steadycellphenotype/issues",
    },
    packages=find_packages(),
    include_package_data=True,
    scripts=["steady_cell_phenotype/scp_converter.py", "start_scp.sh"],
    zip_safe=False,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
)
