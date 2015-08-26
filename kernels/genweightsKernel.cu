#include <kernelDefs.h>
#include <thrust/random.h>
//using
extern __constant__ int params[];
//endofusing

__global__ void genWeightsKern( kernelArray<double> Vec, uint32_t in, size_t device_offset){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ind = params[11] + idx + device_offset; // offset to start of weights, stride of num weights (stride of numWeights)
    thrust::minstd_rand0 randEng;
    randEng.seed(in);
    thrust::uniform_real_distribution<double> uniDist(-1,1);
    randEng.discard(idx);
    Vec.array[ind] = uniDist(randEng);
}


