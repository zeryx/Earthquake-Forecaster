#include <kernelDefs.h>

__global__ void reduceFirstKern(kernelArray<double> Vec, kernelArray<double> per_block_sum, kernelArray<int> params,  size_t device_offset){
    extern __shared__ float sumData[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int fit = params.array[19] + idx + device_offset;

    // load input into __shared__ memory
    double x = 0;
    if(!isnan(Vec.array[fit]))
        x = Vec.array[fit];
    else
        x = 0;

    sumData[threadIdx.x] = x;
    __syncthreads();

    // contiguous range pattern
    for(int offset = blockDim.x / 2;
        offset > 0;
        offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            // add a partial sum upstream to our own
            sumData[threadIdx.x] += sumData[threadIdx.x + offset];
        }
        // wait until all threads in the block have
        // updated their partial sums
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0)
    {
        per_block_sum.array[blockIdx.x] = sumData[0];
    }
}

__global__ void reduceSecondKern(kernelArray<double> per_block_results, kernelArray<int> params, double *result){
    unsigned int idx = threadIdx.x+ blockIdx.x*blockDim.x;
    if(idx ==0){
        *result =0;
        for(int i=0; i<per_block_results.size; i++){
            *result += per_block_results.array[i];

        }
        *result = *result/params.array[10];
    }
}
