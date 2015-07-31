#include "network.h"
#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <ctime>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include <tinyxml2.h>

template <typename T>
__global__ void genWeights( DataArray<T> ref, long in, int nRegWeights, int indLength){
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
                               const int &numOutNeurons, const int &numHiddenLayers,  const std::map<const int, int> &connections){
    this->_NNParams.resize(15, 0); // room to grow
    _NNParams[1] = numInNeurons + numHiddenNeurons + numMemoryNeurons + numOutNeurons;
    _NNParams[2] = numInNeurons + numHiddenNeurons + numOutNeurons;
    _NNParams[3] = numInNeurons;
    _NNParams[4] = numHiddenNeurons;
    _NNParams[5] = numMemoryNeurons;
    _NNParams[6] = numOutNeurons;
    _NNParams[7] = numHiddenLayers;
    _connections = connections;
}

void NetworkGenetic::initializeWeights(){
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL (cudaEventCreate(&start));
    CUDA_SAFE_CALL (cudaEventCreate(&stop));
    int blocksize; //the blocksize defined by the configurator
    int minGridSize; //the minimum grid size needed to achive max occupancy
    int gridSize; // the actual grid size needed
    int individualSize = _NNParams[1]+1;//contains all neruons, plus 1 for fitness vals
    _initialPopsize = _DGeneticsData.size()/(individualSize);
    std::cout<<"initial population: "<<_initialPopsize<<std::endl;
    long time = std::clock();
    cudaEventRecord(start, 0);
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL (cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blocksize, (void*)genWeights<double>, 0, _initialPopsize));
    gridSize = (_initialPopsize + blocksize -1)/blocksize;
    genWeights<double><<<gridSize, blocksize>>>(convertToKernel<double>(_DGeneticsData), time, _NNParams[2], individualSize);
    cudaDeviceSynchronize();
    float miliseconds = 0;
    CUDA_SAFE_CALL (cudaEventRecord(stop, 0));
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL (cudaEventElapsedTime(&miliseconds, start, stop));
    std::cout<<"weight generation: total compute time: "<<miliseconds<<" ms"<<std::endl;
    std::cout<<"effective bandwidth (GB/s) : "<<(_DGeneticsData.size()*8)/((miliseconds/1000)*1e9)<<std::endl;
}



void NetworkGenetic::init(std::string siteInfo){
    int SLEN, sampleRate;
    std::vector<double> sitesData;
   tinyxml2::XMLDocument doc;
   doc.LoadFile(siteInfo.c_str());
   std::string title = doc.FirstChildElement("Sites")->FirstChildElement("site")->GetText();
   std::cout<<"title"<<std::endl;
//    this->_HInitData[0] = doc.first_node()->
//    this->_HInitData[1] = sampleRate;
//    for(int i=0; i<SLEN; i++){
//        this->_HInitData[2+i] = sitesData[i];
//    }
}


void NetworkGenetic::allocateHostAndGPUObjects(unsigned int hostMemory, unsigned int deviceMemory,
                                               std::map<const std::string, float> pHostRam,  std::map<const std::string, float> pDeviceRam){
    unsigned int hostInitDataAlloc = (9*2+2)/sizeof(double); //sampleRate, number of sites, site lon/lat data.
    unsigned int hostGeneticsAlloc = hostMemory*pHostRam.at("genetics")/sizeof(double); //since these are doubles, divide bytes by 8
    unsigned int hostTrainingAlloc = hostMemory*pHostRam.at("input & training")/(sizeof(double)+2);//half for training, half for input I think?
    unsigned int hostInputsAlloc = hostMemory*pHostRam.at("input & training")/(sizeof(float)+2); // their either floats or ints, same amount of bytes.
    unsigned int deviceGeneticsAlloc = deviceMemory*pDeviceRam.at("genetics")/sizeof(double);
    unsigned int deviceTrainingAlloc = deviceMemory*pDeviceRam.at("input & training")/(sizeof(double)+2);
    unsigned int deviceInputsAlloc = deviceMemory*pDeviceRam.at("input & training")/(sizeof(double)+2);
    unsigned int devicePMAIAlloc = 2160*80/sizeof(double); //1.4 mb worth of planetary magnetic activity index for all tests, can store outside of container with other constants.
    //initialize all vectors
    this->_HInitData.resize(hostInitDataAlloc);
    this->_HGeneticsData.resize(hostGeneticsAlloc);
    this->_HTrainingData.resize(hostTrainingAlloc);
    this->_HInputData.resize(hostInputsAlloc);
    this->_DGeneticsData.resize(deviceGeneticsAlloc);
    this->_DTrainingData.resize(deviceTrainingAlloc);
    this->_DInputData.resize(deviceInputsAlloc);
    this->_DPMAIndex.resize(devicePMAIAlloc);
}

void NetworkGenetic::errorFunc(){
    //    cudaEvent_t start, stop;
    //    cudaEventCreate(&start);
    //    cudaEventCreate(&stop);
    //    int blocksize; //the blocksize defined by the configurator
    //    int minGridSize; //the minimum grid size needed to achive max occupancy
    //    int gridSize; // the actual grid size needed
}
