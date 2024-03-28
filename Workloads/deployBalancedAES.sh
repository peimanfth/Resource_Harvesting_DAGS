#!/bin/bash

set -e
memory=64
timeout=600000

cd ParallelAES
dir=$(pwd)

echo -e "\033[0;31mStarting Video Analytics\033[0m"

compose DAG.js > DAG.json
deploy AS DAG.json -w -i -m $memory -t $timeout
../setRedis.sh $dir

echo -e "\033[0;31mStarting Invocation\033[0m"

wsk -i action invoke AS -P input5.json

echo -e "\033[0;31mEnd of Execution\033[0m"

cd ..

