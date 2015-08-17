#include <kernelDefs.h>
#include <thrust/random.h>

__global__ void genWeightsKern( kernelArray<double> ref, uint32_t in, kernelArray<int> params, size_t offset){
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    int ind = params.array[5] + idx; // offset to start of weights, stride of num weights (stride of numWeights)
    thrust::minstd_rand0 randEng;
    thrust::uniform_real_distribution<double> uniDist(0,1);
    randEng.discard(in+idx);
    ref.array[ind] = uniDist(randEng);
}


