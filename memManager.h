#ifndef MEMMANAGER_H
#define MEMMANAGER_H
#include <thrust/device_vector.h>
#include "dataarray.h"
#include <map>
using thrust::device_vector;
class MemManager{
public:
     MemManager();
    bool memoryAlloc(std::map<const std::string, float> pHostRam,
                    std::map<const std::string, float> pDeviceRam,
                    int individualLength, float pMaxHost, float pMaxDevice);
    dataArray<double> genetics();
    dataArray<int> input();
    dataArray<int64_t> training();
    dataArray<int64_t> kpIndex();
    dataArray<int64_t> sites();

public:
    bool geneticsBufferSwap(dataArray<double> *dGen);
    bool GeneticsPushToHost(dataArray<double> *dGen);
    bool InputRefresh(dataArray<int> *input);

public:
    void importSitesData();
    void importKpData();
    void importTrainingData();
    void importGQuakes();

    void setPath(std::string);
    void setTest(int testNum);



private:
    device_vector<double> _DGenetics; //device loaded memory object containing the training weights & fitness data.
    device_vector<int> _DInput; //device loaded memory object containing the site input data.that cannot fit in GPU memory due to resource limitations.
    device_vector<int64_t> _DTraining; //device loaded memory object containing the answer key for all tests (never changes)
    device_vector<int64_t> _DKpIndex; //device loaded memory object containing the planetary magnetic activity index. (all of it should fit)
    device_vector<int64_t> _DSites; //site info (loc, sample rate) update for each test
    device_vector<int64_t> _DGQuakes; //global quakes, updated for each test run
    hVector<double> _HGenetics; //overflow of Genetics Data, updated constantly.
    hVector<int> _HInput; //overflow of input Data, updated constantly.

    long long _deviceGeneticsAlloc;
    long long _deviceInputAlloc;
    long long _hostGeneticsAlloc;
    long long _hostInputAlloc;

    std::string _dataDirectory;
    std::string _testDirectory;
};
#endif
