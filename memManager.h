#ifndef MEMMANAGER_H
#define MEMMANAGER_H
#include <thrust/device_vector.h>
#include "dataarray.h"
#include <map>
using thrust::device_vector;
class MemManager{
public:
     MemManager();
    int memoryAlloc(std::map<const std::string, float> pHostRam,
                    std::map<const std::string, float> pDeviceRam,
                    int individualLength, float pMaxHost, float pMaxDevice);
    dataArray<double> genetics();
    dataArray<int> input();
    dataArray<double> training();
    dataArray<float> kpIndex();
    dataArray<int> init();
    dataArray<double> sites();

public:
    bool geneticsBufferSwap(dataArray<double> *dGen);
    bool GeneticsPushToHost(dataArray<double> *dGen);
    bool InputandTrainingRefresh(dataArray<int> *input,
                                 dataArray<double> *training);

public:
    void importSitesData(std::string);
    void importKpData(std::string);
    void importTrainingData(std::string);
    void importGQuakes(std::string);



private:
    device_vector<double> _DGenetics; //device loaded memory object containing the training weights & fitness data.
    device_vector<int> _DInput; //device loaded memory object containing the site input data.
    device_vector<double> _DTraining; //device loaded memory object containing the teacher data.
    device_vector<float> _DKpIndex; //device loaded memory object containing the planetary magnetic activity index. (all of it should fit)
    device_vector<int>  _DInit; //storage of input from Init function call.
    device_vector<double> _DSites;
    hVector<double> _HGenetics; //overflow of Genetics Data
    hVector<int> _HInput; //overflow of input Data that cannot fit in GPU memory due to resource limitations.
    hVector<double> _HTraining; //overflow of training data

    long _deviceGeneticsAlloc;
    long _deviceInputAlloc;
    long _deviceTrainingAlloc;
    long _hostGeneticsAlloc;
    long _hostInputAlloc;
    long _hostTrainingAlloc;

};
#endif
