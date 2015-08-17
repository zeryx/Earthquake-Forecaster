#include <kernelDefs.h>

__global__ void reduceKern(kernelArray<double> weights,kernelArray<double> per_block_results,kernelArray<int> params, int n, int device_offset, int blockOffset){
    extern __shared__ float sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ind = idx*params.array[7] + device_offset;

    // load input into __shared__ memory
    float x = 0;
    if(idx < n)
    {
        x = weights.array[ind + params.array[2]];
    }
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
        per_block_results.array[blockIdx.x+blockOffset] = sdata[0];
    }
}
