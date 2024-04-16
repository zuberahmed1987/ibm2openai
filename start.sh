#!/bin/bash

export DIRNAME=$(dirname $0)

cd ${DIRNAME}

. ./venv/bin/activate
. .env
#exec python ibmapi.py
exec uvicorn ibmapi:app --host 0.0.0.0 --port 5050 --workers 4
