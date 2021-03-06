# SteadyCellPhenotype

A tool for the computation of steady states and dynamics of intracellular biological networks.

## Current operation

The site is implemented in Python 3 using the packages `flask`, `matplotlib`, `networkx`, `numba`, and `pathos`. Some of these may be installed by default if you have installed the standard anaconda distribution. Others need to use `pip` to install it. If the version of `python` on your system is version 3 (check using `python --version`) you can install these dependencies using
```
python -m pip install flask matplotlib networkx numba pathos
```
commonly, such as on Mac or Linux-based systems, Python 3.x is installed as `python3` and you will need to run
```
python3 -m pip install flask matplotlib networkx numba pathos
```


Then, on mac or linux, the site can then be started by 
```
./start.sh
```
Assuming that you are in the top directory of this project. This will open up a web server on the loopback address on port 5000. If that's gibberish, what I
mean is that
* It won't be accessible to the broader internet and
* On the same machine, you can point your web browser at
  [http://localhost:5000](http://localhost:5000) and access the site.

The flask documentation contains info on how to get the thing working for remote users.  [Flask
documentation](https://flask.palletsprojects.com/en/1.1.x/)

# `convert.py` command-line usage

Running `convery.py --help` displays information about command line options 

```
usage: convert.py [-h] [-i INPUTFILE] [-o OUTPUTFILE] [-n] [-no-polys] [-sim] [-graph] [-init-val INIT_VAL [INIT_VAL ...]] [--count COUNT] [-c]
                  [-comit CONTINUOUS_OMIT [CONTINUOUS_OMIT ...]] [-power SELF_POWER]

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
  -init-val INIT_VAL [INIT_VAL ...]
                        for simulators, fix initial values got some variables Ex: -init-val LIP 1
  --count COUNT         number of random points tried by the simulator, default 1,000,000. Ignored if the -sim flag is not used
  -c, --continuous      generate polynomials for continuous system, applied before the self-power operation
  -comit CONTINUOUS_OMIT [CONTINUOUS_OMIT ...], --continuous-omit CONTINUOUS_OMIT [CONTINUOUS_OMIT ...]
                        list of variables to _not_ apply continuity operation to
  -power SELF_POWER, --self-power SELF_POWER
                        gets polynomials for a power of the system. i.e. self-composition, power-1 times (default: 1) ignored for simulator Warning: This can take a long
                        time!
```

## Notes:

[Here](http://www2.macaulay2.com/Macaulay2/Downloads/GNU-Linux/Debian/index.html) is where you can
get Debian packages for Macaulay2. 
