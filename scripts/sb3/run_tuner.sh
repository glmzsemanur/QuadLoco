#!/bin/bash


for i in $(seq 1 500)
do
   echo "Starting Trial $i of 500"
   # Run the script. It will run 1 trial and exit.
   python tuner2.py --headless
   
   # Optional: sleep for a few seconds to let the GPU cool down
   sleep 2
done