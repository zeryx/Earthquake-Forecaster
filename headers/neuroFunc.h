#ifndef NEUROFUNC_H
#define NEUROFUNC_H
#include <cuda_runtime.h>

 __host__ __device__ void neuroSum(double &store, double &input);

 __host__ __device__ void neuroZero(double &store);

 __host__ __device__ void neuroSquash(double &store);

 __host__ __device__ void neuroMemGate(double &memIn, double &input, double &output, double min);

 __host__ __device__ void neuroMemForget(double &memForget, double &mem, double min);


#endif
