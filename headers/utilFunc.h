#ifndef UTILFUNC_H
#define UTILFUNC_H
#include <cuda_runtime.h>
//functions

__host__ __device__ float bearingCalc(float lat1, float lon1, float lat2, float lon2);

__host__ __device__ float distCalc(float lat1, float lon1, float lat2, float lon2);

__host__ __device__ float normalize(float x, float mean, float stdev);

__host__ __device__ double shift(double x, double oldMax, double oldMin, double newMax, double newMin);

__host__ __device__ double ActFunc(double x);

__host__  __device__ double scoreFunc(double whenGuess, float whenAns, double latGuess,
                                      double lonGuess, double latAns, double lonAns, double avgFit, float certainty);





#endif
