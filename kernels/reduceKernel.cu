#include <kernelDefs.h>

__global__ void reduceFirstKern(kernelArray<double> weights,kernelArray<double> per_block_results, kernelArray<int> params,  int device_offset){
    extern __shared__ float sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ind = params.array[19] + idx + device_offset;

    // load input into __shared__ memory
    double x = 0;

    x = weights.array[ind];

    sdata[threadIdx.x] = x;
    __syncthreads();

    // contiguous range pattern
    for(int offset = blockDim.x / 2;
        offset > 0;
        offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            // add a partial sum upstream to our own
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        // wait until all threads in the block have
        // updated their partial sums
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0)
    {
        sdata[0] = sdata[0] / blockDim.x;
        per_block_results.array[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceSecondKern(kernelArray<double> per_block_results, kernelArray<int> params, double *result){
    unsigned int idx = threadIdx.x+ blockIdx.x*blockDim.x;
    if(idx ==0){
        *result =0;
        for(int i=0; i<per_block_results.size; i++){
            *result += per_block_results.array[i];

        }
        *result = *result/per_block_results.size;
    }
}
