#include <iostream>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

struct GenRand
{
    __device__ float operator () (int idx)
    {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist;
        randEng.discard(idx);
        return uniDist(randEng);
    }
};

int main()
{
//    int *d_a, *d_b; //Device variable Declaration

//        //Allocation of Device Variables
//    cudaMalloc((void **)&d_a, sizeof(int));
//    cudaMalloc((void **)&d_b, sizeof(int));

//        //Copy Host Memory to Device Memory
//    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
thrust::device_vector<float> rvect(1000);
thrust::transform(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(1000),
                  rvect.begin(),
                  GenRand());

for(int i=0; i<rvect.size(); i++){
    std::cout<<rvect[i]<<std::endl;
}

        //Launch Kernel
//    AddIntsCUDA << <1, 1 >> >(d_a, d_b);

//        //Copy Device Memory to Host Memory
//    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

//    printf("The answer is %d",a);


//        //Free Device Memory
//        cudaFree(d_a);
//    cudaFree(d_b);

    return 0;
}
