#include <kernelDefs.h>

__global__ void transferKern(kernelArray<double> vect, kernelArray<int> pos, kernelArray<int> params, int ind_offset, size_t device_offset){
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int wt_offset = params.array[11] + device_offset;
    const int fit_offset = params.array[19] + device_offset;
    int i=0;
    for(i=0; i<params.array[10]; i++){
        if(pos.array[i+ind_offset] == idx)
            break;
    }
    double tmp_fit = vect.array[fit_offset+idx];
    vect.array[fit_offset+idx] = vect.array[fit_offset+i];
    vect.array[fit_offset+i] = tmp_fit;
    for(int j=0; j<params.array[1]; j++){
        double tmp_wt = vect.array[wt_offset+idx];
        vect.array[wt_offset+idx] = vect.array[wt_offset+i];
        vect.array[wt_offset+i] = tmp_wt;
    }
}
