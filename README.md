# SteadyCellPhenotype

A tool for the computation of steady states and exploration of dynamics in intracellular biological networks.

## Developers and Research Team

Adam Knapp (Developer), Julia Chifman (PI), Luis Sordo-Vieira (Co-PI), Reinhard Laubenbacher (Co-PI)

Funding by College of Arts and Sciences Mellon Fund, American University. (2019)  

## Background on Ternary models and Examples

See https://steadycellphenotype.github.io/docs.html

## Installation and Operation

The app is implemented in Python 3 with the help of a variety of packages including `flask`, `matplotlib`, `networkx`, `numba`, `pathos`, and `attrs`. Some of these may be installed by default if you have installed the standard anaconda distribution. Others need to use `pip` to install it. If the version of `python` on your system is version 3 (check using `python --version`) you can install these dependencies using
```
python -m pip install -r requirements.txt
```
from the `steadycellphenotype` directory. Commonly, such as on Mac or Linux-based systems, Python 3.x is installed as `python3` and you will need to run
```
python3 -m pip install -r requirements.txt
```

Then, on Mac or linux, the site can then be started by 
```
./start_scp.sh
```
You can also install SteadyCellPhenotype as a python package via pip:
```
python3 -m pip install -e .
```
This will add two executables to your path, `start_scp.sh` and `scp_converter.py`. You can also install SteadyCellPhenotype in a virtual environment, see below, which may help with package versioning issues. 

Assuming that you are in the top directory of this project. This will open up a web server on the loopback address on port 5000. If that's gibberish, what I mean is that
* It won't be accessible to the broader internet and
* On the same machine, you can point your web browser at
  [http://localhost:5000](http://localhost:5000) and access the site.

The flask documentation contains info on how to get the thing working for remote users.  [Flask
documentation](https://flask.palletsprojects.com/en/1.1.x/)

### Note:

Instructions on obtaining Macaulay2 are [here](http://www2.macaulay2.com/Macaulay2/Downloads/). In particular, [here](http://www2.macaulay2.com/Macaulay2/Downloads/GNU-Linux/Debian/index.html) is where you can get Debian packages. 


## Using a virtual environment

If you encounter compatibility errors between various python and package versions on your computer, it may be useful to create a virtual environment with project-specific versions. (e.g. as of this writing there are issues with the current versions of numba and python 3.9 on macOS.) To create a virtual environment, first find the desired version of python on your machine. (We assume 3.8 below.) _Hint_: On macOS and Linux, we can use
```
find / -name python3.8 2>&1 | grep -v "Permission denied"
```
to find all copies of the python 3.8 executable on the system. Then, using the desired copy of python, run
```
/whatever/path/to/python/you/found/python3.8 -m venv stdy-cll-phntyp-venv
```
replacing `stdy-cll-phntyp-venv` with whatever folder name you desire the virtual environment to reside in. (Often we choose `venv` inside of the source directory.) All packages which we install will reside in this directory. Then whenever we want to enter the virtual environment, we run
```
source stdy-cll-phntyp-venv/bin/activate
```
Now run
```
python3 -m pip install -e .
```

# Command line tool, `scp_converter.py`

In addition to the browser interface, SteadyCellPhenotype provides a command line tool: `scp_converter.py` for advanced users. This tool allows the user to perform various transformations to a MAX/MIN/NOT/polynomial model, including:
* generation of a pure-polynomial model, (i.e. conversion of MAX/MIN/NOT formulae to the corresponding polynomial)
* conversion of some or all formulae to the corresponding continuous version, 
* generation of a "self-power" of the system, i.e. ![$F^n = \underbrace{F \circ \cdots \circ F}_{n}$](./images/math1.svg) suitable for computing cycles as fixed points of the composed system.
* generation of several C language programs derived from the model:
  * a simulator which runs the system on a random sample of initial conditions, searching for attractors
  * a simulator which does a complete state space search of the system, searching for attractors
  * a simulator which creates a graph representation of the update function on state space
  
  Each of these generated programs can be compiled using recent versions of `gcc` and require the header files (`*.h`) included in the `steady_cell_phenotype` directory. Output is in JSON format.

Commands line options for `scp_converter.py` are shown by running `scp_converter.py --help` which displays:

```
usage: scp_converter.py [-h] [-i INPUTFILE] [-o OUTPUTFILE] [-n] [-no-polys] [-sim] [-graph] [-complete_search] [-init-val INIT_VAL [INIT_VAL ...]]
                        [--count COUNT] [-c] [-comit CONTINUOUS_OMIT [CONTINUOUS_OMIT ...]] [-power SELF_POWER]

Converter from MAX/MIN/NOT formulae to either low-degree polynomials over F_3 or a C-language simulator

optional arguments:
  -h, --help            show this help message and exit
  -i INPUTFILE, --inputfile INPUTFILE
                        input filename containing MAX/MIN/NOT formulae. required.
  -o OUTPUTFILE, --outputfile OUTPUTFILE
                        output filename for the polynomial formulae. if not provided, stdout is used
  -n, --non_descriptive
                        use non-descriptive names for variables
  -no-polys             do not output polynomials, used by default when output is by simulator
  -sim                  output C-language simulator program
  -graph                use the graph-creation simulator
  -complete_search      completely search the state-space
  -init-val INIT_VAL [INIT_VAL ...]
                        for simulators, fix initial values got some variables Ex: -init-val LIP 1
  --count COUNT         number of random points tried by the simulator, default 1,000,000. Ignored if the -sim flag is not used
  -c, --continuous      generate polynomials for continuous system, applied before the self-power operation
  -comit CONTINUOUS_OMIT [CONTINUOUS_OMIT ...], --continuous-omit CONTINUOUS_OMIT [CONTINUOUS_OMIT ...]
                        list of variables to _not_ apply continuity operation to
  -power SELF_POWER, --self-power SELF_POWER
                        gets polynomials for a power of the system. i.e. self-composition, power-1 times (default: 1) ignored for simulator Warning: This can
                        take a long time!
```

[![Python package](https://github.com/knappa/steadycellphenotype/actions/workflows/python-package.yml/badge.svg)](https://github.com/knappa/steadycellphenotype/actions/workflows/python-package.yml)

[![Upload Python Package](https://github.com/knappa/steadycellphenotype/actions/workflows/python-publish.yml/badge.svg)](https://github.com/knappa/steadycellphenotype/actions/workflows/python-publish.yml)