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
__global__ void genWeightsKern( kernelArray<double> ref, size_t in, kernelArray<int> params, size_t offset);

__global__ void NetKern(kernelArray<double> Vec, kernelArray<int> params, Order* connections, int hour, kernelArray<double> meanCh,
                        kernelArray<double> stdCh, size_t device_offset);

__global__ void reduceFirstKern(kernelArray<double> Vec,
                                kernelArray<double> per_block_results,
                                kernelArray<int> params, size_t device_offset);

__global__ void reduceSecondKern(kernelArray<double> per_block_results, kernelArray<int> params, double *result);

__global__ void normalizeKern(kernelArray<double> Vec, kernelArray<int> params, double *avgFitness, size_t device_offset);

__global__ void evolutionKern(kernelArray<double> vect, kernelArray<int> params, int *childOffset, int *realGridSize, size_t in, size_t device_offset);

__global__ void bitonicSortKern(kernelArray<double> Vec, kernelArray<int> params, int j, int k, size_t device_offset);

__global__ void cutoffKern(kernelArray<double>vect, kernelArray<int> params, int *childOffset, int *evoGridSize, double *avgFitness, size_t device_offset);
#endif
