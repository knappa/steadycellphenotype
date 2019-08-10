#!/bin/sh

export FLASK_APP=steady_cell_phenotype
#export FLASK_ENV=development

# so that it is visible to all:
#flask run --host=0.0.0.0

# only on localhost
flask run
