#ifndef NEUROFUNC_H
#define NEUROFUNC_H
#include <cuda_runtime.h>

 __host__ __device__ void neuroSum(double &store, double &input);

 __host__ __device__ void neuroMulti(double &store, const double &input);

 __host__ __device__ void neuroZero(double &store);

 __host__ __device__ void neuroSquash(double &store);

 __host__ __device__ void neuroMemGateOut(double &memGate, double &input, double &output);

 __host__ __device__ void neuroMemGateIn(double &memGate, double &input, double &output);

 __host__ __device__ void neuroMemGateForget(double &memForget, double &mem);


#endif
