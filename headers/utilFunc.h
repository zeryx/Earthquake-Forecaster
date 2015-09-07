#ifndef UTILFUNC_H
#define UTILFUNC_H
#include <cuda_runtime.h>
//functions

__host__ __device__ float bearingCalc(float lat1, float lon1, float lat2, float lon2);

__host__ __device__ float distCalc(float lat1, float lon1, float lat2, float lon2);

__host__ __device__ float normalize(float x, float mean, float stdev);

__host__ __device__ float shift(float x, float max, float min);

__host__ __device__ float ActFunc(float x);

__host__  __device__ float scoreFunc(float whenGuess, int whenAns, float latGuess, float lonGuess, float latAns, float lonAns);





#endif
