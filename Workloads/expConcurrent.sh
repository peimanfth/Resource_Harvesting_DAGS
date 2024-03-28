#!/bin/bash

set -e
memory=64
timeout=600000

cd ParallelAES
dir=$(pwd)

echo -e "\033[0;31mStarting Video Analytics\033[0m"

compose DAGB.js > DAGB.json
compose DAGC.js > DAGC.json
deploy ASB DAGB.json -w -i -m $memory -t $timeout
deploy ASC DAGC.json -w -i -m $memory -t $timeout
../setRedis.sh $dir

echo -e "\033[0;31mStarting Concurrent Invocation\033[0m"

wsk -i action invoke ASB -P input2.json
wsk -i action invoke ASC -P input2.json

echo -e "\033[0;31mEnd of Invocation\033[0m"

cd ..

