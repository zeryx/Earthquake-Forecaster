#ifndef NETWORK_H
#define NETWORK_H
#include "memManager.h"
#include <string>
#include <thrust/pair.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
class  NetworkGenetic{
public:
    NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons,
                   const int &numOutNeurons,const int &numHiddeLayers, const thrust::pair<int, int> &connections);
    void errorFunc();
    void initializeWeights(); //initializes _data array and fills with random numbers
    void getTestInfo(std::string dataFolder); // gets Kp, GQuakes, SiteInfo xml doc data and puts it in the gpu.
    void allocateHostAndGPUObjects(std::map<const std::string, float> pHostRam,
                                   std::map<const std::string, float> pDeviceRam,
                                   float pMaxHost, float pMaxDevice);
private:
    MemManager _memVirtualizer; // component that handles memory virtualization and transfer
    thrust::pair<int, int> _connections;
    thrust::device_vector<int> _NNParams; // only vector that stays on here
    dataArray<double> _genetics;
    dataArray<int> _input;
    dataArray<Answers> _training;
    dataArray<Kp> _kpIndex;
    dataArray<SiteInfo> _sites;
};


#endif
