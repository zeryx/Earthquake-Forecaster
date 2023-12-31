#include <kernelDefs.h>

__global__ void cutoffKern(kernelArray<double> Vec, kernelArray<int> params, int *childOffset, int *evoGridSize, double *avgFitness, size_t device_offset){
    const int idx = blockIdx.x * blockDim.x +threadIdx.x;
    const int fitnessval = params.array[19] + idx + device_offset;
    if(Vec.array[fitnessval] > 0 && Vec.array[fitnessval+1] == 0){
        *childOffset = idx;
        *evoGridSize = (params.array[10]-*childOffset)/512;
    }
}
