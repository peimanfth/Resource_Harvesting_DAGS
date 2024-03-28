#!/bin/bash

set -e
memory=1
timeout=240000

# cd video-analytics

# echo -e "\033[0;31mStarting Video Analytics\033[0m"

# compose DAG.js > DAG.json
# deploy vid DAG.json -w -i

# echo -e "\033[0;31mStarting Invocation\033[0m"

# wsk -i action invoke vid -P input.json

# echo -e "\033[0;31mEnd of Invocation\033[0m"

# cd ..

cd ml-pipeline
# dir=$(pwd)

echo -e "\033[0;31mStarting ML Pipeline\033[0m"

compose DAG.js > DAG.json
deploy ml DAG.json -w -i -m $memory -t $timeout

cd ../../front-end/inputs
dir=$(pwd)
file="ml.json"
../setRedis.sh $dir $file
# echo -e "\033[0;31mStarting Invocation\033[0m"

# wsk -i action invoke ml -P input.json

# echo -e "\033[0;31mEnd of Invocation\033[0m"

cd ..

# cd mapreduce

# echo -e "\033[0;31mStarting Map Reduce\033[0m"

# compose DAG.js > DAG.json
# deploy mr DAG.json -w -i

# echo -e "\033[0;31mStarting Invocation\033[0m"

# wsk -i action invoke mr -P input.json

# echo -e "\033[0;31mEnd of Invocation\033[0m"

# cd ..