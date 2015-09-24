#!/bin/bash
trap "exit" INT
sync | sudo tee /proc/sys/vm/drop_caches
java -jar tester.jar -exec "nvprof -o timeline.nvprof ./Earthquake_Forecaster" -folder "../mount/data/" -seed 113 -train
java -jar tester.jar -exec "nvprof -o timeline.nvprof ./Earthquake_Forecaster" -folder "../mount/data/" -seed 142 -train
java -jar tester.jar -exec "nvprof -o timeline.nvprof ./Earthquake_Forecaster" -folder "../mount/data/" -seed 129 -train
java -jar tester.jar -exec "nvprof -o timeline.nvprof ./Earthquake_Forecaster" -folder "../mount/data/" -seed 148 -train
java -jar tester.jar -exec "nvprof -o timeline.nvprof ./Earthquake_Forecaster" -folder "../mount/data/" -seed 151 -train
java -jar tester.jar -exec "nvprof -o timeline.nvprof ./Earthquake_Forecaster" -folder "../mount/data/" -seed 156 -train
java -jar tester.jar -exec "nvprof -o timeline.nvprof ./Earthquake_Forecaster" -folder "../mount/data/" -seed 169 -train
