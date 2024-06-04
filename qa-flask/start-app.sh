#!/bin/sh
echo $SCRIPT_NAME
echo $HOST:$PORT
SCRIPT_NAME=$SCRIPT_NAME gunicorn -b $HOST:$PORT app:app