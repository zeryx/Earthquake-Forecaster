#include "network.h"
#include "dataarray.h"
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <ctime>
#include <thrust/host_vector.h>
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
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    thrust::minstd_rand0 randEng;
    randEng.seed(idx);
    long seed = idx+ref._size*in;
    for(int i=0; i<nRegWeights; i++){
        thrust::uniform_real_distribution<double> uniDist(0,1);
        randEng.discard(seed);
        ref._array[idx*indLength + i] = uniDist(randEng);
    }
}


NetworkGenetic::NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons,
                               const int &numOutNeurons, const int &numHiddenLayers,
                               const thrust::pair<int, int> &connections){
    this->_NNParams.resize(15, 0); // room to grow
    _NNParams[1] = numInNeurons + numHiddenNeurons + numMemoryNeurons + numOutNeurons;
    _NNParams[2] = numInNeurons + numHiddenNeurons + numOutNeurons;
    _NNParams[3] = numInNeurons;
    _NNParams[4] = numHiddenNeurons;
    _NNParams[5] = numMemoryNeurons;
    _NNParams[6] = numOutNeurons;
    _NNParams[7] = numHiddenLayers;
    _NNParams[8] = numInNeurons + numHiddenLayers + numMemoryNeurons + numOutNeurons + 1 + 1; //1 for fitness, 1 for community output composite vector
    _connections = connections;
}

void NetworkGenetic::initializeWeights(){
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL (cudaEventCreate(&start));
    CUDA_SAFE_CALL (cudaEventCreate(&stop));
    std::cout<<"about to initialize weights"<<std::endl;
    int blocksPerGrid; //the blocksize defined by the configurator
    int threadsblock = 512; // the actual grid size needed
    int seedItr = 0;
    do{
        _NNParams[9] = _genetics._size/(_NNParams[8]); // number of individuals on device.
        long seed = std::clock() + std::clock()*seedItr++;
        std::cout<<"seed: "<<seed<<std::endl;
        blocksPerGrid=(_NNParams[9]+threadsblock-1)/threadsblock;
        genWeights<double><<<blocksPerGrid, threadsblock>>>(_genetics, seed, _NNParams[2], _NNParams[8]);
        cudaDeviceSynchronize();
    }while(_memVirtualizer.GeneticsPushToHost(&_genetics));
    _NNParams[9] = _genetics._size/(_NNParams[8]); // number of individuals on device.
    long seed = std::clock() + std::clock()*seedItr++;
    std::cout<<"seed: "<<seed<<std::endl;
    blocksPerGrid=(_NNParams[9]+threadsblock-1)/threadsblock;
    genWeights<double><<< blocksPerGrid, threadsblock>>>(_genetics, seed, _NNParams[2], _NNParams[8]);
    cudaDeviceSynchronize();
}


void NetworkGenetic::allocateHostAndGPUObjects(std::map<const std::string, float> pHostRam,
                                               std::map<const std::string, float> pDeviceRam,
                                               float pMaxHost, float pMaxDevice){
    _memVirtualizer.memoryAlloc(pHostRam, pDeviceRam, _NNParams[8], pMaxHost, pMaxDevice);
    _genetics = _memVirtualizer.genetics();
    _input = _memVirtualizer.input();
    _sites = _memVirtualizer.sites();
    _kpIndex = _memVirtualizer.kpIndex();
    _training = _memVirtualizer.training();
}

void NetworkGenetic::getTestInfo(std::string dataFolder){
    _memVirtualizer.setPath(dataFolder);
    _memVirtualizer.setTest(105);
    _memVirtualizer.importSitesData();
    _memVirtualizer.importKpData();
    _memVirtualizer.importGQuakes();
    _memVirtualizer.importTrainingData();
}

void NetworkGenetic::errorFunc(){
    //    cudaEvent_t start, stop;
    //    cudaEventCreate(&start);
    //    cudaEventCreate(&stop);
    //    int blocksize; //the blocksize defined by the configurator
    //    int minGridSize; //the minimum grid size needed to achive max occupancy
    //    int gridSize; // the actual grid size needed
}
