#!/bin/bash

set -e
memory=42
timeout=240000

echo -e "\033[0;31mStarting Compilation\033[0m"

cd video-analytics

echo -e "\033[0;31mStarting Video Analytics\033[0m"

cd video-streaming

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update streaming --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

cd ../../

cd decoder

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update decoder --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

cd ../../

cd image-recognition

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update recognition1 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p index 0

cd ../../

cd image-recognition

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update recognition2 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p index 1

cd ../../


#ml-pipeline

mem1=1024
mem2=2048
mem3=3072
mem4=4096

cd ../ml-pipeline
echo -e "\033[0;31mStarting ML Pipeline\033[0m"

cd PCA

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update pca --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

cd ../../

cd ParamTune

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update paramtune1 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p index 0

cd ../../

cd ParamTune

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update paramtune2 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p index 1

cd ../../

cd ParamTune

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update paramtune3 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p index 2

cd ../../

cd ParamTune

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update paramtune4 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p index 3

cd ../../

cd combine

rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update combine --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

cd ../../


#ml-pipeline

cd ../mapreduce
echo -e "\033[0;31mStarting Map Reduce\033[0m"

cd partitioner
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update partitioner --kind python:3.10 --main main --memory 4096 --timeout $timeout index.zip

cd ../../

cd mapper
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update mapper0 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p mapperId 0

cd ../../

cd mapper
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update mapper1 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p mapperId 1

cd ../../

cd mapper
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update mapper2 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip -p mapperId 2

cd ../../

cd reducer
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update reducer --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

cd ../../

# waitALU

cd ../waitALU
echo -e "\033[0;31mStarting WaitALU\033[0m"

cd waitInput
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update waitInput --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

cd ../../

cd Alu
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update ALU --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

cd ../../

#ParallelAES

mem1=1024
mem2=2048
mem3=2560

cd ../ParallelAES
echo -e "\033[0;31mStarting ParallelAES\033[0m"

cd wait1
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update wait1 --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

cd ../../

cd AES
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update AES1 --kind python:3.10 --main main --memory $mem3 --timeout $timeout index.zip -p index 1

cd ../../

cd AES
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update AES2 --kind python:3.10 --main main --memory $mem2 --timeout $timeout index.zip -p index 2

cd ../../

cd AES
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *


wsk -i action update AES3 --kind python:3.10 --main main --memory $mem1 --timeout $timeout index.zip -p index 3

cd ../../

cd Stats
rm -rf build
mkdir build

cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update Stats --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

cd ../../




echo -e "\033[0;31mFinished Compilation\033[0m"