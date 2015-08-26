#include <kernelDefs.h>
//using
extern __constant__ int params[];
//endofusing

__global__ void findChildrenKern(kernelArray<double>Vec, int *childOffset, double* avgFitness, size_t device_offset){
    const int idx = blockIdx.x * blockDim.x +threadIdx.x;
    const int fitnessval = params[19] + idx + device_offset;
    if(Vec.array[fitnessval] < *avgFitness && Vec.array[fitnessval-1] > *avgFitness)
        *childOffset = idx;
}
