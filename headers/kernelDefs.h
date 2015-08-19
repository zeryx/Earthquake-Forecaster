#ifndef KERNELDEFS_H
#define KERNELDEFS_H
#include <cuda_runtime.h>
#include "dataarray.h"
#include <utility>
extern __constant__ int input[40*20*3];
extern __constant__ int site_offset[20];
extern __constant__ int channel_offset[3];
//functions
__host__ __device__ double bearingCalc(double lat1, double lon1, double lat2, double lon2);

__host__ __device__ double distCalc(double lat1, double lon1, double lat2, double lon2);

__host__ __device__ double normalize(double x, double mean, double stdev);

__host__ __device__ double shift(double x, double max, double min);

__host__ __device__ double ActFunc(double x);

__host__ __device__ double cosd(double x);

__host__ __device__ double sind(double x);

//main kernels
__global__ void genWeightsKern( kernelArray<double> ref, uint32_t in, kernelArray<int> params, size_t offset);

__global__ void NetKern(kernelArray<double> Vec, kernelArray<int> params, kernelArray<double> globalQuakes,
                     kernelArray<double> siteData, kernelArray<double> answers,
                    kernelArray<std::pair<int, int> > connections, double Kp, int numOfSites,
                    int hour, kernelArray<double> meanCh, kernelArray<double> stdCh, size_t device_offset);

__global__ void reduceKern(kernelArray<double> weights,
                                kernelArray<double> per_block_results,
                                kernelArray<int> params, int device_offset);

__global__ void evoKern(kernelArray<double> weights, kernelArray<int> params, int device_offset);


//utility kernels
__global__ void interKern(kernelArray<int> in, kernelArray<int> out, int* site_offset, int* channel_offset,  int sampleRate, int numOfSites);



#endif
