#include <kernelDefs.h>
#include <thrust/random.h>

__global__ void genWeightsKern( kernelArray<double> ref, uint32_t in, kernelArray<int> params, size_t offset){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ind = idx*params.array[3]+offset;
    thrust::minstd_rand0 randEng;
    thrust::uniform_real_distribution<double> uniDist(0,1);
    for(int i=0; i<params.array[2]; i++){
        randEng.discard(in+ind);
        ref.array[ind+i] = uniDist(randEng);
    }
    for(int i=params.array[2]; i<params.array[3]; i++){
        ref.array[ind+i]=0;
    }
}


