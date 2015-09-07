#ifndef NEUROFUNC_H
#define NEUROFUNC_H
#include <cuda_runtime.h>

 __host__ __device__ void neuroSum(float &store, float &input);

 __host__ __device__ void neuroZero(float &store);

 __host__ __device__ void neuroSquash(float &store);

 __host__ __device__ void neuroMemGate(float &memIn, float &input, float &output, float min);

 __host__ __device__ void neuroMemForget(float &memForget, float &mem, float min);


#endif
