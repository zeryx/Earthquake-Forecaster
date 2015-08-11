#ifndef NETWORK_H
#define NETWORK_H
#include "memManager.h"
#include "dataarray.h"
#include <vector>
#include <string>
#include <map>
#include <thrust/pair.h>
#include <cuda.h>
#include "stdlib.h"
#include <cuda_runtime_api.h>
class  NetworkGenetic{
public:
    NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons,
                   const int &numOutNeurons, const int &numWeights,  std::vector< thrust::pair<int, int> >&connections);
    void errorFunc();
    void initializeWeights(); //initializes _data array and fills with random numbers
    void allocateHostAndGPUObjects(float pMax, size_t deviceRam, size_t hostRam);
    bool init(int sampleRate, int SiteNum, std::vector<double>siteData);
    void doingTraining(int site, int hour, double lat,
                       double lon, double mag, double dist);
    void forecast(std::vector<double> *ret, int& hour, std::vector<int> *data, double &K, std::vector<double> *globalQuakes);
    void storeWeights(std::string filepath);
    bool checkForWeights(std::string filepath);
private:
    unifiedArray<double> device_genetics;
    unifiedArray<double> host_genetics;
    unifiedArray<double> host_genetics_device;
    std::vector<thrust::pair<int, int> > *_connect;
    thrust::device_vector<int>_NNParams; // only vector that stays on here
    thrust::device_vector<double> _siteData;
    thrust::device_vector<double> _answers;
    std::vector<double> _best;
    bool _istraining;
    int _sampleRate;
    int _numofSites;
    size_t _numOfStreams;
    size_t _streambytes;
    size_t _streamSize;
    std::vector<cudaStream_t> _stream;
};


#endif
