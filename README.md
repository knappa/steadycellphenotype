# steadycellphenotype

Computations of steady states of intracellular biological network

## Work in progress

A work in progress, so many things are not in their final state. But it now does _something_, so it
is worth putting it on github.

There is still plenty to do here.

partial TODO list: 
* macauly integration
* quickstart
* documentation
* docker stuff

## Current operation

The site is implemented in Python 3 using the package `flask`. This is installed by default if you
have installed the standard anaconda distribution. Others, like me, need to use `pip` to
install it. In that case:
```
pip install flask
```
or on some Linux-based systems (certainly on Debian) you will need to run
```
pip3 install flask
```


The site can then be started up by running `site.py` by
```
python site.py
```
or
```
python3 site.py
```

This will open up a web server on the loopback address on port 5000. If that's gibberish, what I
mean is that
* It won't be accessible to the broader internet and
* On the same machine, you can point your web browser at
  [http://localhost:5000](http://localhost:5000) and access the site.


The flask documentation contains info on how to get the thing working for remote users.  [Flask
documentation](https://flask.palletsprojects.com/en/1.1.x/)
