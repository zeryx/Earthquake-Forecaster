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
extern __constant__ int params[];
extern __constant__ int site_offset[];
extern __constant__ int channel_offset[];
extern __constant__ int trainingsize;
//functions
__host__ __device__ float bearingCalc(float lat1, float lon1, float lat2, float lon2);

__host__ __device__ float distCalc(float lat1, float lon1, float lat2, float lon2);

__host__ __device__ float normalize(float x, float mean, float stdev);

__host__ __device__ float shift(float x, float max, float min);

__host__ __device__ float ActFunc(float x);

__host__ __device__ float scoreFunc(float whenGuess, float whenAns, int hour, float latGuess, float lonGuess, float latAns, float lonAns);



//mutations union definition


//main kernels
__global__ void genWeightsKern( kernelArray<double> ref, uint32_t in,  size_t offset);

__global__ void NetKern(kernelArray<double> Vec,   kernelArray<std::pair<const int, const int> > connections,
                        int hour, kernelArray<double> meanCh, kernelArray<double> stdCh, size_t device_offset);

__global__ void reduceFirstKern(kernelArray<double> Vec,kernelArray<double> per_block_results,size_t device_offset);

__global__ void reduceSecondKern(kernelArray<double> per_block_results,  double *result);

__global__ void normalizeKern(kernelArray<double> Vec,  double *avgFitness, size_t device_offset);

__global__ void evolutionKern(kernelArray<double> Vec,  int *childOffset, uint32_t in, size_t device_offset);

__global__ void bitonicSortKern(kernelArray<double> Vec,  int j, int k, size_t device_offset);

__global__ void findChildrenKern(kernelArray<double>Vec,  int *childOffset, double *avgFitness, size_t device_offset);
#endif
