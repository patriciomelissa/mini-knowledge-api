#!/bin/bash

python3 -m venv env-mini-knowledge-api
source env/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

pre-commit install