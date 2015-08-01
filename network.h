#ifndef NETWORK_H
#define NETWORK_H
#include "dataarray.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <map>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
class  NetworkGenetic{
public:
    NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons,
                   const int &numOutNeurons,const int &numHiddeLayers, const std::map<const int, int> &connections);
    void errorFunc();
    void initializeWeights(); //initializes _data array and fills with random numbers
    void allocateHostAndGPUObjects(unsigned int hostMemory, unsigned int deviceMemory,
                                   std::map<const std::string, float> pHostRam,  std::map<const std::string, float> pDeviceRam);
    void init(std::string siteInfo);
private:
    thrust::device_vector<int> _NNParams;//1.
    thrust::device_vector<double> _DGeneticsData; //device loaded memory object containing the training weights & fitness data.
    thrust::device_vector<int> _DInputData; //device loaded memory object containing the site input data.
    thrust::device_vector<double> _DTrainingData; //device loaded memory object containing the teacher data.
    thrust::device_vector<double> _DPMAIndex; //device loaded memory object containing the planetary magnetic activity index. (all of it should fit)
    thrust::device_vector<int>  _DInitData; //storage of input from Init function call.
    thrust::device_vector<double> _DSitesData;
    thrust::host_vector<double> _HGeneticsData; //overflow of Genetics Data
    thrust::host_vector<int> _HInputData; //overflow of input Data that cannot fit in GPU memory due to resource limitations.
    thrust::host_vector<double> _HTrainingData; //overflow of training data
    std::map<const int, int> _connections;
};


#endif
