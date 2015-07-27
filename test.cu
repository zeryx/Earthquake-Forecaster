#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime_api.h>

__global__ void AddIntsCUDA(int *a, int *b) //Kernel Definition
{
    *a = *a + *b;
}

int main()
{
    int a = 5, b = 9;
    int *d_a, *d_b; //Device variable Declaration

        //Allocation of Device Variables
    cudaMalloc((void **)&d_a, sizeof(int));
    cudaMalloc((void **)&d_b, sizeof(int));

        //Copy Host Memory to Device Memory
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);


        //Launch Kernel
    AddIntsCUDA << <1, 1 >> >(d_a, d_b);

        //Copy Device Memory to Host Memory
    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The answer is ",a);


        //Free Device Memory
        cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
