import os

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()


def prerelease_local_scheme(version):
    """Return local scheme version unless building on master in Gitlab.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on Gitlab for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if "CIRCLE_BRANCH" in os.environ and os.environ["CIRCLE_BRANCH"] == "master":
        return ""
    else:
        return get_local_node_and_date(version)


setup(
    name="steady_cell_phenotype",
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
    setup_requires=["setuptools_scm"],
    use_scm_version={"local_scheme": prerelease_local_scheme},
)
