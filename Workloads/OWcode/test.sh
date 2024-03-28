#!/bin/bash
#print "Starting Test" in red
echo -e "\e[31mStarting Test\e[0m"
wsk -i action update pyact pyact.py
wsk -i action update hi hello.py
wsk -i action update bye bye.py
wsk -i action update wait wait.py

# run pyact.py as wsk action with input 1
wsk -i action invoke pyact --param number 3
wsk -i action invoke pyact --param number 1
wsk -i action invoke pyact --param number 2
wsk -i action invoke bye
wsk -i action invoke wait --param time 10
wsk -i action invoke pyact --param number 4
wsk -i action invoke hi
wsk -i action invoke pyact --param number 1
sleep 3
wsk -i action invoke pyact --param number 3
wsk -i action invoke pyact --param number 4
wsk -i action invoke wait --param time 3
wsk -i action invoke hi
wsk -i action invoke pyact --param number 2

echo -e "\e[31mTest Complete\e[0m"