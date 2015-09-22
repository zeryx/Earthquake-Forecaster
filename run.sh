#!/bin/bash
trap "exit" INT
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
java -jar tester.jar -exec "nvprof -o timeline.nvprof ./Earthquake_Forecaster" -folder "../mount/data/" -seed 113 -train
