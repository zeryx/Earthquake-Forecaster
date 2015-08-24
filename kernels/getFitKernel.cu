#include <kernelDefs.h>


__global__ void getFitKern(kernelArray<double> in, kernelArray<int> params, kernelArray<std::pair<int, double> > out, size_t device_offset, int ind_offset){
    const int idx =  threadIdx.x + blockDim.x*blockIdx.x;
    const int fit = params.array[19]+idx+device_offset;
    out.array[idx+ind_offset].first = idx; // store the original individual number as well.
    out.array[idx+ind_offset].second = in.array[fit];//fitness is stored in array at pos 1

}
