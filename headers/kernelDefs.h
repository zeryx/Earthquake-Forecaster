#ifndef KERNELDEFS_H
#define KERNELDEFS_H
#include <cuda_runtime.h>
#include "dataarray.h"
#include <utility>

//functions
__host__ __device__ double bearingCalc(double lat1, double lon1, double lat2, double lon2);

__host__ __device__ double distCalc(double lat1, double lon1, double lat2, double lon2);

__host__ __device__ double normalize(double x, double mean, double stdev);

__host__ __device__ double shift(double x, double max, double min);

__host__ __device__ double ActFunc(double x);

__host__ __device__ double cosd(double x);

__host__ __device__ double sind(double x);

//kernels
__global__ void genWeightsKern( kernelArray<double> ref, uint32_t in, kernelArray<int> params, size_t offset);

__global__ void NetKern(kernelArray<double> weights, kernelArray<int> params, kernelArray<double> globalQuakes,
                    kernelArray<int> inputVal, kernelArray<double> siteData, kernelArray<double> answers,
                    kernelArray<std::pair<int, int> > connections, double Kp, int sampleRate,int numOfSites,
                    int hour, double meanCh1, double meanCh2, double meanCh3, double stdCh1, double stdCh2,
                    double stdCh3, size_t offset);

__global__ void reduceKern(kernelArray<double> weights,
                                kernelArray<double> per_block_results,
                                kernelArray<int> params, int device_offset, int blockOffset);

__global__ void evoKern(kernelArray<double> weights, kernelArray<int> params, int device_offset);





#endif
