#include <kernelDefs.h>

__global__ void getFitKern(kernelArray<double> in, kernelArray<int> params, kernelArray<double> fit, kernelArray<int> pos, int ind_offset, size_t device_offset){
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    pos.array[idx+ind_offset] = idx;
    fit.array[idx+ind_offset] = in.array[params.array[19] + idx + device_offset];
}
