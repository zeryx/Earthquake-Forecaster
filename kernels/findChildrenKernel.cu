#include <kernelDefs.h>

__global__ void findChildrenKern(kernelArray<double>vect, kernelArray<int> params, int *childOffset, double* avgFitness, size_t device_offset){
    const int idx = blockIdx.x * blockDim.x +threadIdx.x;
    const int fitnessval = params.array[19] + idx + device_offset;
    if(vect.array[fitnessval] < *avgFitness && vect.array[fitnessval-1] > *avgFitness)
        *childOffset = idx;
}
