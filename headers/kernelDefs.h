#ifndef KERNELDEFS_H
#define KERNELDEFS_H
#include <cuda_runtime.h>
#include "dataarray.h"
#include <connections.h>
#include <utility>
#include <thrust/pair.h>
extern __constant__ int inputData[];
extern __constant__ double answers[];
extern __constant__ double globalQuakes[];
extern __constant__ double siteData[];
extern __constant__ double Kp;
extern __constant__ int site_offset[];
extern __constant__ int channel_offset[];
extern __constant__ int trainingsize;



//main kernels
__global__ void genWeightsKern( kernelArray<float> Vec, size_t in, kernelArray<int> params, size_t offset);

__global__ void NetKern(kernelArray<float> Vec, kernelArray<int> params, Order* connections, int hour, kernelArray<double> meanCh,
                        kernelArray<double> stdCh, size_t device_offset);

__global__ void reduceFirstKern(kernelArray<float> Vec,
                                kernelArray<float> per_block_results,
                                kernelArray<int> params, size_t device_offset);

__global__ void reduceSecondKern(kernelArray<float> per_block_results, kernelArray<int> params, float *result);

__global__ void normalizeKern(kernelArray<float> Vec, kernelArray<int> params, float *avgFitness, size_t device_offset);

__global__ void evolutionKern(kernelArray<float> Vect, kernelArray<int> params, int *childOffset, int *realGridSize, size_t in, size_t device_offset);

__global__ void bitonicSortKern(kernelArray<float> Vec, kernelArray<int> params, int j, int k, size_t device_offset);

__global__ void cutoffKern(kernelArray<float> Vec, kernelArray<int> params, int *childOffset, int *evoGridSize, float *avgFitness, size_t device_offset);
#endif
