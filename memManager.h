#ifndef MEMMANAGER_H
#define MEMMANAGER_H
#include <thrust/device_vector.h>
#include <fstream>
#include "dataarray.h"
using thrust::device_vector;
class MemManager{
public:
     MemManager();
    bool memoryAlloc(int individualLength, float pMaxHost, float pMaxDevice);
    dataArray<double> genetics();

public:
//    bool geneticsBufferSwap(dataArray<double> *dGen);
//    bool GeneticsPushToHost(dataArray<double> *dGen);
    void initFromStream(std::ifstream& stream);
    void pushToStream(std::string filename);
    hVector<double> _HGenetics;
    device_vector<double> _DGenetics; //device loaded memory object containing the training weights & fitness data.
private:


    long long _deviceGeneticsAlloc;
    long long _hostGeneticsAlloc;

};
#endif
