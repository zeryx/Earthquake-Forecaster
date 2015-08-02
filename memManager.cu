#include "memManager.h"
#include "getsys.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/swap.h>
#include <thrust/system_error.h>
#include "tinyxml2.h"
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

//xml error message handling
#ifndef XMLCheckResult
#define XMLCheckResult(a_eResult) if (a_eResult != tinyxml2::XML_SUCCESS) { printf("Error: %i\n", a_eResult);  exit(a_eResult); }
#endif

MemManager::MemManager(){}


dataArray<double> MemManager::genetics(){
    return convertToKernel(_DGenetics);
}

dataArray<int> MemManager::init(){
    return convertToKernel(_DInit);
}

dataArray<int> MemManager::input(){
    return convertToKernel(_DInput);
}

dataArray<double> MemManager::training(){
    return convertToKernel(_DTraining);
}

dataArray<double> MemManager::sites(){
    return convertToKernel(_DSites);
}

dataArray<float> MemManager::kpIndex(){
    return convertToKernel(_DKpIndex);
}



int MemManager::memoryAlloc(std::map<const std::string, float> pHostRam,
                            std::map<const std::string, float> pDeviceRam,
                            int individualLength, float pMaxHost, float pMaxDevice){
    long hostMem = GetHostRamInBytes()*pMaxHost; //make a host memory container, this is the max
    long deviceMem = GetDeviceRamInBytes()*pMaxDevice; //dito for gpu
    int dub = 8, integer = 4;
    _hostGeneticsAlloc = hostMem*pHostRam.at("genetics")/dub; //since these are doubles, divide bytes by 8
    _hostTrainingAlloc = hostMem*pHostRam.at("input & training")/(dub*2);//half for training, half for input I think?
    _hostInputAlloc = hostMem*pHostRam.at("input & training")/(integer*2); // their either floats or ints, same amount of bytes.
    _deviceGeneticsAlloc = deviceMem*pDeviceRam.at("genetics")/dub;
    _deviceTrainingAlloc = deviceMem*pDeviceRam.at("input & training")/(dub*2);
    _deviceInputAlloc = deviceMem*pDeviceRam.at("input & training")/(integer*2);
    //round the genetics allocators to whole individuals.
    _hostGeneticsAlloc = (_hostGeneticsAlloc/individualLength)*individualLength;
    _deviceGeneticsAlloc = (_deviceGeneticsAlloc/individualLength)*individualLength;
    long hostAlloced = _hostGeneticsAlloc*dub + _hostTrainingAlloc*dub + _hostInputAlloc*integer;
    long deviceAlloced = _deviceGeneticsAlloc*dub + _deviceTrainingAlloc*dub + _deviceInputAlloc*integer;
    std::cout<<hostAlloced<<std::endl;
    std::cout<<deviceAlloced<<std::endl;
    //initialize all large vectors (everything not from an xml file)
    try{
    this->_HGenetics.setMax(_hostGeneticsAlloc);
    this->_HTraining.setMax(_hostTrainingAlloc);
    this->_HInput.setMax(_hostInputAlloc);
    }
    catch(thrust::system_error &e){
        std::cerr<<"Error resizing vector Element: "<<e.what()<<std::endl;
        exit(1);
    }
    catch(std::bad_alloc &e){
        std::cerr<<"Ran out of space due to : "<<"host"<<std::endl;
        std::cerr<<e.what()<<std::endl;
        std::cout<<GetHostRamInBytes()<<std::endl;
        exit(1);
    }
    try{
        this->_DGenetics.resize(_deviceGeneticsAlloc);
        this->_DTraining.resize(_deviceTrainingAlloc);
        this->_DInput.resize(_deviceInputAlloc);
    }
    catch(thrust::system_error &e){
        std::cerr<<"Error resizing vector Element: "<<e.what()<<std::endl;
        exit(1);
    }
    catch(std::bad_alloc &e){
        std::cerr<<"Ran out of space due to : "<<"device"<<std::endl;
        std::cerr<<e.what()<<std::endl;
        std::cout<<GetDeviceRamInBytes()<<std::endl;
        exit(1);
    }
    return true;
}
bool MemManager::geneticsBufferSwap(dataArray<double> *dGen){
    return false;
}

bool MemManager::GeneticsPushToHost(dataArray<double> *dGen){
    long dGenLength = dGen->_size;
    if(_HGenetics._itr + dGenLength*2 > _HGenetics._maxLen){ //if _HGenetics can take 2 more at the current size, keep going
        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics._hVect.begin()+_HGenetics._itr);
        _HGenetics._itr = _HGenetics._itr + dGenLength; //set the iterator  to the new position.
        return true;
    }
    else if(_HGenetics._itr + dGenLength*2 <= _HGenetics._maxLen){//if _HGenetics can only take 1 or exactly 2 at current size, resize dgen to fit
        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics._hVect.begin()+_HGenetics._itr);
        _HGenetics._itr = _HGenetics._itr + dGenLength;
        _DGenetics.resize(_HGenetics._maxLen - _HGenetics._itr);// the device_vector for genetics was resized to fit the remaining host mem container.
        dGen->_size = _DGenetics.size();
        return true;
    }
    else if(_HGenetics._itr+dGenLength == _HGenetics._maxLen){//if the _HGenetics vector is full, tell the GPU
        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics._hVect.begin()+_HGenetics._itr);
        _HGenetics._itr = 0;
        _DGenetics.resize(_deviceGeneticsAlloc);
        dGen->_size =_DGenetics.size();
        return false;
    }
    else{ // not sure how you got here
        std::cerr<<"how did you get here? not sure!"<<std::endl;
        return false;
    }
}

void MemManager::importSitesData(std::string siteInfo){
    int dataSet, SLEN;
    tinyxml2::XMLDocument doc;
    if(_DInit.size()>0){ //empty any previous data located in array, both are small enough to be of no consquence
    _DInit.clear();
    _DInit.shrink_to_fit();
    }
    if(_DSites.size()>0){
    _DSites.clear();
    _DSites.shrink_to_fit();
    }
    doc.LoadFile(siteInfo.c_str());
    tinyxml2::XMLNode * pRoot = doc.FirstChild();
    if(pRoot == NULL) exit(tinyxml2::XML_ERROR_FILE_READ_ERROR);
    tinyxml2::XMLElement * pElement = pRoot->NextSiblingElement("Sites");
    if(pElement == NULL) exit(tinyxml2::XML_ERROR_PARSING_ELEMENT);
    tinyxml2::XMLError eResult = pElement->QueryIntAttribute("data_set", &dataSet);
    XMLCheckResult(eResult);

    eResult = pElement->QueryIntAttribute("num_sites", &SLEN);
    XMLCheckResult(eResult);
    _DInit.push_back(SLEN);
    _DInit.push_back(dataSet);
    tinyxml2::XMLElement *SitesList = pRoot->NextSiblingElement("Site");

    while(SitesList != NULL){
        int sampleData;
        double longitude, latitude;
        eResult = SitesList->QueryIntAttribute("sample_rate", &sampleData);
        XMLCheckResult(eResult);
        _DInit.push_back(sampleData);
        eResult = SitesList->QueryDoubleAttribute("latitude", &latitude);
        XMLCheckResult(eResult);
        _DSites.push_back(latitude);
        eResult = SitesList->QueryDoubleAttribute("longitude", &longitude);
        XMLCheckResult(eResult);
        _DSites.push_back(longitude);
        SitesList = SitesList->NextSiblingElement("Site");
    }
}

void MemManager::importKpData(std::string Kp){
            tinyxml2::XMLDocument doc;
        tinyxml2::XMLError eResult;
        _DKpIndex.clear();
        _DKpIndex.shrink_to_fit();
        doc.LoadFile(Kp.c_str());
        tinyxml2::XMLNode *pRoot = doc.FirstChild();
        if(pRoot == NULL) exit(tinyxml2::XML_ERROR_FILE_READ_ERROR);
        tinyxml2::XMLElement * pElement = pRoot->NextSiblingElement("Kp");
        if(pElement == NULL) exit(tinyxml2::XML_ERROR_PARSING_ELEMENT);
        tinyxml2::XMLElement * KpList = pElement->FirstChildElement("Kp_hr");
        while(KpList != NULL){
            int seconds;
            float magnitude;
            eResult = KpList->QueryIntAttribute("secs", &seconds);
            XMLCheckResult(eResult);
            _DKpIndex.push_back(seconds);
            eResult = KpList->QueryFloatText(&magnitude);
            XMLCheckResult(eResult);
            _DKpIndex.push_back(magnitude);
            KpList = KpList->NextSiblingElement("Kp_hr");
    }

}
