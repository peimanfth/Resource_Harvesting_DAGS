#!/bin/bash

set -e


# Accept index as a command line argument
if [ -z "$1" ]; then
  echo "Index not provided. Usage: $0 <index>"
  exit 1
fi

index=$1
# Default memory in MB
base_memory=512
# Memory for other functions in MB
other_memory=128
timeout=360000
# memory1=$(($base_memory))
memory1=$(($base_memory * 1))
echo $memory1
# memory2=$(($base_memory * $index * 2))
memory2=$(($base_memory * 2))
echo $memory2
# memory3=$(($base_memory * $index * 4))
memory3=$(($base_memory * 3))
echo $memory3



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
# for aes_module in AES1 AES2 AES3; do
#   aes_index=${aes_module: -1}
#   memory=$((base_memory * index * aes_index))
cd AES
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update AES1 --kind python:3.10 --main main --memory $memory1 --timeout $timeout index.zip -p index 1

cd ../../

cd AES
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update AES2 --kind python:3.10 --main main --memory $memory2 --timeout $timeout index.zip -p index 2

cd ../../

cd AES
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update AES3 --kind python:3.10 --main main --memory $memory3 --timeout $timeout index.zip -p index 3

cd ../../
# done

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
