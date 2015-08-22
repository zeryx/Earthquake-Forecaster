#ifndef KERNELDEFS_H
#define KERNELDEFS_H
#include <cuda_runtime.h>
#include "dataarray.h"
#include <utility>
#include <thrust/pair.h>
extern __constant__ int input[];
extern __constant__ double answers[];
extern __constant__ double globalQuakes[];
extern __constant__ double siteData[];
extern __constant__ double Kp;
extern __constant__ int site_offset[];
extern __constant__ int channel_offset[];
extern __constant__ int trainingsize;

//functions
__host__ __device__ float bearingCalc(float lat1, float lon1, float lat2, float lon2);

__host__ __device__ float distCalc(float lat1, float lon1, float lat2, float lon2);

__host__ __device__ float normalize(float x, float mean, float stdev);

__host__ __device__ float shift(float x, float max, float min);

__host__ __device__ float ActFunc(float x);


//mutations union definition
union mutations{
    char c[8];
    int f[2];
    double result;
};

//main kernels
__global__ void genWeightsKern( kernelArray<double> ref, uint32_t in, kernelArray<int> params, size_t offset);

__global__ void NetKern(kernelArray<double> Vec, kernelArray<int> params,  kernelArray<std::pair<const int, const int> > connections,
                        int numOfSites,int hour, kernelArray<double> meanCh, kernelArray<double> stdCh,
                        size_t device_offset);

__global__ void reduceFirstKern(kernelArray<double> weights,
                                kernelArray<double> per_block_results,
                                kernelArray<int> params, int device_offset);

__global__ void reduceSecondKern(kernelArray<double> per_block_results, double *result);

__global__ void evoFirstKern(kernelArray<double> weights, kernelArray<int> params, float avgFitness, int device_offset);

__global__ void evoSecondKern(kernelArray<double> weights, kernelArray<int> params, int device_offset, Lock lock);


//utility kernels



#endif
