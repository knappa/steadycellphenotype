from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='steady_cell_phenotype',
    version='0.1.0',
    author="A.C. Knapp",
    author_email="adam.knapp@instanton.org",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knappa/steadycellphenotype",
    project_urls={
        "Bug Tracker": "https://github.com/knappa/steadycellphenotype/issues",
        },
    packages=find_packages(),
    include_package_data=True,
    scripts=['steady_cell_phenotype/scp_converter.py', 'start_scp.sh'],
    zip_safe=False,
    install_requires=[
        'flask>=1.1',
        'matplotlib>=3.4',
        'networkx>=2.5',
        'numba>=0.53',
        'pathos>=0.2.7',
        'attrs>=20.3',
        'Werkzeug>=1',
        'numpy>=1.20'
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    python_requires=">=3.7",
    )
