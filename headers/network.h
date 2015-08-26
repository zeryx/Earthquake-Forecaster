#ifndef NETWORK_H
#define NETWORK_H
#include <dataarray.h>
#include <vector>
#include <string>
#include <map>
#include <cuda.h>
#include <utility>
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do{cudaError_t err = call; if (cudaSuccess != err) {fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",__FILE__, __LINE__, cudaGetErrorString(err) ); cudaDeviceReset(); exit(EXIT_FAILURE);}} while (0)
#endif

class  NetworkGenetic{
public:
    NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons,
                   const int &numOutNeurons, const int &numWeights,  std::vector< std::pair<int, int> >&connections);
    void errorFunc();
    void generateWeights(); //initializes _data array and fills with random numbers
        void setParams();
    void allocateHostAndGPUObjects(float pMax, size_t deviceRam, size_t hostRam);
    bool init(int sampleRate, int SiteNum, std::vector<double> *siteData);
    void doingTraining(int site, int hour, double lat,
                       double lon, double mag, double dist);
    void forecast(std::vector<double> *ret, int& hour, std::vector<int> *data, double &K, std::vector<double> *globalQuakes);
    void reformatTraining(std::vector<int>* old_input, std::vector<double> ans, std::vector<double>* sitedata, std::vector<double>* globalquakes, double kp);
    void storeWeights(std::string filepath);
    bool checkForWeights(std::string filepath);
    void sort();
private:
    kernelArray<double> device_genetics;
    kernelArray<double> host_genetics;
    std::vector<std::pair<int, int> > *_connect;
    std::vector<double> *_siteData;
    std::vector<double> _answers;
    std::vector<double> _best;
    kernelArray<int>_hostParams;
    bool _istraining;
    int _numOfStreams;
    size_t _streambytes;
    int _streamSize;
    int *_channel_offset;
    int *_site_offset;
    std::vector<cudaStream_t> _stream;
};


#endif
