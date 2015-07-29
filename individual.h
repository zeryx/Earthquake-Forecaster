#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H
#include <thrust/device_vector.h>
class Individual{//stores the weights and its fitness values
public:
    __device__ __host__ Individual();
     thrust::device_vector<double> _weights;
    __device__ __host__ bool setAbsFitness(double newAbsFitness);
    __device__ __host__ double absFitness();//gets the absolute fitness value
    __device__ __host__ bool calcRelativeFitness(double averageAbsFitness);//calculate the relative fitness
    __device__ __host__ double relativeFitness();//gets the normalized fitness value from all individuals
private:
    double _absoluteFitness, _relativeFitness;
};


#endif
