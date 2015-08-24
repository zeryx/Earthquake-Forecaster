#include <kernelDefs.h>

__global__ void findChildrenKern(kernelArray<double>vect, kernelArray<int> params, size_t *childOffset, size_t device_offset){
    const int idx = blockIdx.x * blockDim.x +threadIdx.x;
    const int fitnessval = params.array[19] + idx + device_offset;
    if(vect.array[fitnessval] == 0 && vect.array[fitnessval-1] !=0){
        *childOffset = idx;
    }
}
