#!/bin/bash

set -e

# Default memory in MB
base_memory=33
# Memory for other functions in MB
other_memory=1
timeout=360000

# Accept index as a command line argument
if [ -z "$1" ]; then
  echo "Index not provided. Usage: $0 <index>"
  exit 1
fi

index=$1

# Calculate memory for ParallelAES based on the index
memory=$((base_memory / index))
echo $memory
echo $memory
echo $memory

echo -e "\033[0;31mStarting Compilation\033[0m"

# Update ParallelAES memory allocation
cd ./ParallelAES
echo -e "\033[0;31mStarting ParallelAES\033[0m"

cd wait1
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update wait1 --kind python:3.10 --main main --memory $other_memory --timeout $timeout index.zip

cd ../../

# Update AES actions with a fixed memory allocation
for aes_module in AES1 AES2 AES3; do
  aes_index=${aes_module: -1}
  cd AES
  rm -rf build
  mkdir build

  cp -R src/* build
  cd build
  zip -r index.zip *

  wsk -i action update $aes_module --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p index $aes_index

  cd ../../
done

# Update Stats with a fixed memory allocation
cd Stats
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update Stats --kind python:3.10 --main main --memory $other_memory --timeout $timeout index.zip

cd ../../

echo -e "\033[0;31mFinished Compilation\033[0m"
