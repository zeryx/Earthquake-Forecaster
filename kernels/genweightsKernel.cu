#include <kernelDefs.h>
#include <thrust/random.h>

__global__ void genWeightsKern( kernelArray<double> ref, uint32_t in, kernelArray<int> params, size_t device_offset){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ind = params.array[11] + idx + device_offset; // offset to start of weights, stride of num weights (stride of numWeights)
    thrust::minstd_rand0 randEng;
    randEng.seed(in);
    thrust::uniform_real_distribution<double> uniDist(-1,1);
    randEng.discard(idx);
    ref.array[ind] = uniDist(randEng);
}


