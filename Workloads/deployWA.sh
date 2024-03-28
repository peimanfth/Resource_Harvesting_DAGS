#!/bin/bash

set -e
memory=64
timeout=120000

cd waitALU
dir=$(pwd)

echo -e "\033[0;31mStarting WaitALU\033[0m"

compose DAG.js > DAG.json
deploy WA DAG.json -w -i -m $memory -t $timeout
../setRedis.sh $dir

echo -e "\033[0;31mStarting Invocation\033[0m"

wsk -i action invoke WA -P input.json

echo -e "\033[0;31mEnd of Invocation\033[0m"

cd ..