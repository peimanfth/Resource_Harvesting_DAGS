#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "No input provided. Usage: $0 <input_memory>"
    exit 1
fi

# Input from the user
input_memory=$1
index=$2
base_memory=128
timeout=360000

# Calculations based on input
memory1=$((input_memory / 3))
echo $memory1
#convert memory1 to int
# memory1=$((memory1))
# Adjust mem1, mem2, mem3 relative to the new memory1
mem1=$((input_memory * (33 - $index) / 100))
echo $mem1
mem2=$((input_memory * 33 / 100))
echo $mem2
mem3=$((input_memory * (34 + $index) / 100))
echo $mem3


echo -e "\033[0;31mStarting Compilation\033[0m"







#ParallelAES


cd ./ParallelAES
echo -e "\033[0;31mStarting ParallelAES\033[0m"

cd wait1
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update wait1_B --kind python:3.10 --main main --memory $base_memory --timeout $timeout index.zip
wsk -i action update wait1_C --kind python:3.10 --main main --memory $base_memory --timeout $timeout index.zip

cd ../../

cd AES
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update AES1_B --kind python:3.10 --main main --memory $memory1 --timeout $timeout index.zip -p index 1
wsk -i action update AES1_C --kind python:3.10 --main main --memory $mem1 --timeout $timeout index.zip -p index 1


cd ../../

cd AES
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update AES2_B --kind python:3.10 --main main --memory $memory1 --timeout $timeout index.zip -p index 2
wsk -i action update AES2_C --kind python:3.10 --main main --memory $mem2 --timeout $timeout index.zip -p index 2

cd ../../

cd AES
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *


wsk -i action update AES3_B --kind python:3.10 --main main --memory $memory1 --timeout $timeout index.zip -p index 3
wsk -i action update AES3_C --kind python:3.10 --main main --memory $mem3 --timeout $timeout index.zip -p index 3

cd ../../

cd Stats
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update Stats_B --kind python:3.10 --main main --memory $base_memory --timeout $timeout index.zip
wsk -i action update Stats_C --kind python:3.10 --main main --memory $base_memory --timeout $timeout index.zip

cd ../../




echo -e "\033[0;31mFinished Compilation\033[0m"