#!/bin/sh

export FLASK_APP=steady_cell_phenotype
#export FLASK_ENV=development

# so that it is visible to all:
#flask run --host=0.0.0.0

# only on localhost
flask run

# the "proper way" for deployment
# pip3 install waitress
# waitress-serve --call 'steady_cell_phenotype:create_app'
