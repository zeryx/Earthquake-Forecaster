#include "network.h"
#include "dataarray.h"
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tinyxml2.h>

//macros
//cuda error message handling
#define CUDA_SAFE_CALL(call)                                          \
    do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
    fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
    __FILE__, __LINE__, cudaGetErrorString(err) );       \
    exit(EXIT_FAILURE);                                           \
    }                                                                 \
    } while (0)


template <typename T>
__global__ void genWeights( dataArray<T> ref, long in, int nRegWeights, int indLength){
    long idx = blockDim.x*blockIdx.x + threadIdx.x;
    long seed= idx+in;
    thrust::default_random_engine randEng;
    for(int i=0; i<nRegWeights; i++){
        thrust::uniform_real_distribution<double> uniDist(0,1);
        randEng.discard(seed);
        ref._array[idx*indLength+i] = uniDist(randEng);
    }
}




NetworkGenetic::NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons,
                               const int &numOutNeurons, const int &numHiddenLayers,
                               const thrust::pair<int, int> &connections, std::string dataFolder){
    this->_NNParams.resize(15, 0); // room to grow
    _NNParams[1] = numInNeurons + numHiddenNeurons + numMemoryNeurons + numOutNeurons;
    _NNParams[2] = numInNeurons + numHiddenNeurons + numOutNeurons;
    _NNParams[3] = numInNeurons;
    _NNParams[4] = numHiddenNeurons;
    _NNParams[5] = numMemoryNeurons;
    _NNParams[6] = numOutNeurons;
    _NNParams[7] = numHiddenLayers;
    _NNParams[8] = numInNeurons + numHiddenLayers + numMemoryNeurons + numOutNeurons + 1 + 9; //1 for fitness, 9 for community output
    _connections = connections;
}

void NetworkGenetic::initializeWeights(){
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL (cudaEventCreate(&start));
    CUDA_SAFE_CALL (cudaEventCreate(&stop));
    int blocksize; //the blocksize defined by the configurator
    int minGridSize; //the minimum grid size needed to achive max occupancy
    int gridSize; // the actual grid size needed
    int cumulative = 0;
    _NNParams[9] = _genetics._size/(_NNParams[8]); // number of individuals on device.
    do{
        _NNParams[9] = _genetics._size/(_NNParams[8]); // number of individuals on device.
        cumulative = cumulative + _NNParams[9];
        std::cout<<"population on device: "<<_NNParams[9]<<std::endl;
        std::cout<<"cumulative population: "<<cumulative<<std::endl;
        CUDA_SAFE_CALL (cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blocksize, (void*)genWeights<double>, 0, _NNParams[9]));
        gridSize = (_NNParams[9] + blocksize -1)/blocksize;
        long seed = std::clock();
        genWeights<double><<<gridSize, blocksize>>>(_genetics, seed, _NNParams[2], _NNParams[8]);
        cudaDeviceSynchronize();
    }while(_memVirtualizer.GeneticsPushToHost(&_genetics));
}



void NetworkGenetic::allocateHostAndGPUObjects(std::map<const std::string, float> pHostRam,
                                               std::map<const std::string, float> pDeviceRam,
                                               float pMaxHost, float pMaxDevice){
    _memVirtualizer.memoryAlloc(pHostRam, pDeviceRam, _NNParams[8], pMaxHost, pMaxDevice);
    _genetics = _memVirtualizer.genetics();
    _input = _memVirtualizer.input();
    _init = _memVirtualizer.init();
    _sites = _memVirtualizer.sites();
    _kpIndex = _memVirtualizer.kpIndex();
    _training = _memVirtualizer.training();
}

void NetworkGenetic::errorFunc(){
    //    cudaEvent_t start, stop;
    //    cudaEventCreate(&start);
    //    cudaEventCreate(&stop);
    //    int blocksize; //the blocksize defined by the configurator
    //    int minGridSize; //the minimum grid size needed to achive max occupancy
    //    int gridSize; // the actual grid size needed
}
