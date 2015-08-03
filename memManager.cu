#include "memManager.h"
#include "getsys.h"
#include "datediff.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
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

dataArray<int> MemManager::input(){
    return convertToKernel(_DInput);
}

dataArray<int64_t> MemManager::training(){
    return convertToKernel(_DTraining);
}

dataArray<int64_t> MemManager::sites(){
    return convertToKernel(_DSites);
}

dataArray<int64_t> MemManager::kpIndex(){
    return convertToKernel(_DKpIndex);
}



bool MemManager::memoryAlloc(std::map<const std::string, float> pHostRam,
                             std::map<const std::string, float> pDeviceRam,
                             int individualLength, float pMaxHost, float pMaxDevice){//allocates memory for genetis & input vectors
    long long hostMem = GetHostRamInBytes()*pMaxHost; //make a host memory container, this is the max
    long long deviceMem = GetDeviceRamInBytes()*pMaxDevice; //dito for gpu
    int dub = 8, integer = 4;
    _hostGeneticsAlloc = hostMem*pHostRam.at("genetics")/dub; //since these are doubles, divide bytes by 8
    _hostInputAlloc = hostMem*pHostRam.at("input & training")/integer; // their either floats or ints, same amount of bytes.
    _deviceGeneticsAlloc = deviceMem*pDeviceRam.at("genetics")/dub;
    _deviceInputAlloc = deviceMem*pDeviceRam.at("input & training")/(integer*2); // a half of the alloced input is goign to XML data
    //round the genetics allocators to whole individuals.
    _hostGeneticsAlloc = (_hostGeneticsAlloc/individualLength)*individualLength;
    _deviceGeneticsAlloc = (_deviceGeneticsAlloc/individualLength)*individualLength;
    std::cout<<"allocating..."<<std::endl;
    //initialize all large vectors (everything not from an xml file)
    try{
        this->_HGenetics.setMax(_hostGeneticsAlloc);
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
    std::cout<<"allocated."<<std::endl;
    return true;
}
bool MemManager::geneticsBufferSwap(dataArray<double> *dGen){
    return false;
}

bool MemManager::GeneticsPushToHost(dataArray<double> *dGen){
    long long dGenLength = dGen->_size;
    long long currpos = dGenLength + _HGenetics._itr;
    std::cout<<"length of device vector: "<<dGenLength<<std::endl;
    std::cout<<"host vector max length: "<<_HGenetics._maxLen<<std::endl;
    if(currpos*2 <= _HGenetics._maxLen && currpos < _HGenetics._maxLen){ //if _HGenetics can take 2 more at the current size, keep going
        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics._hVect.begin()+_HGenetics._itr);
        _HGenetics._itr = currpos; //set the iterator  to the new position.
        std::cout<<"#1"<<std::endl;
        return true;
    }
    else if(currpos*2 > _HGenetics._maxLen && currpos < _HGenetics._maxLen){//if _HGenetics can only take 1 or exactly 2 at current size, resize dgen to fit
        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics._hVect.begin()+_HGenetics._itr);
        _DGenetics.resize(_HGenetics._maxLen - currpos);// the device_vector for genetics was resized to fit the remaining host mem container.
        std::cout<<"resized genetics to: "<<_HGenetics._maxLen - _HGenetics._itr<<std::endl;
        dGen->_size = _DGenetics.size();
        _HGenetics._itr = currpos;
        std::cout<<"#2"<<std::endl;
        return true;
    }
    else if(currpos == _HGenetics._maxLen && dGenLength !=0){//if the _HGenetics vector is full, tell the GPU
        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics._hVect.begin()+_HGenetics._itr);
        _HGenetics._itr = 0;
        _DGenetics.resize(_deviceGeneticsAlloc);
        dGen->_size =_DGenetics.size();
        std::cout<<"#3"<<std::endl;
        return false;
    }
    else if(currpos > _HGenetics._maxLen){
        std::cout<<"#4"<<std::endl;
        return false;
    }

    return false;
}


bool MemManager::InputRefresh(dataArray<int> *input){
    return false;
}
void MemManager::setPath(std::string pathToData){
    this->_dataDirectory = pathToData;
}

void MemManager::setTest(int testNum){
    _testDirectory = _dataDirectory;
    std::ostringstream oss;
    oss << "/" << testNum;
    _testDirectory.append(oss.str());

}

void MemManager::importSitesData(){
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError eResult;
    _DSites.clear();
    _DSites.shrink_to_fit();
    std::string siteInfoStr = this->_testDirectory;
    siteInfoStr.append("/SiteInfo.xml");
    eResult = doc.LoadFile(siteInfoStr.c_str());
    XMLCheckResult(eResult);
    tinyxml2::XMLNode * pRoot = doc.FirstChild();
    if(pRoot == NULL){
        std::cout<<"file read error"<<std::endl;
        exit(tinyxml2::XML_ERROR_FILE_READ_ERROR);
    }
    tinyxml2::XMLElement * pElement = pRoot->NextSiblingElement("Sites");
    if(pElement == NULL){
        std::cout<<"error parsing element"<<std::endl;
        exit(tinyxml2::XML_ERROR_PARSING_ELEMENT);
    }
    tinyxml2::XMLElement *SitesList = pRoot->NextSiblingElement("Site");

    while(SitesList != NULL){
        int sampleRate, siteNumber;
        float longitude, latitude;
        eResult = SitesList->QueryIntAttribute("sample_rate", &sampleRate);
        XMLCheckResult(eResult);
        eResult = SitesList->QueryFloatAttribute("latitude", &latitude);
        XMLCheckResult(eResult);
        eResult = SitesList->QueryFloatAttribute("longitude", &longitude);
        XMLCheckResult(eResult);
        eResult = SitesList->QueryIntText(&siteNumber);
        XMLCheckResult(eResult);
        _DSites.push_back(siteNumber);
        _DSites.push_back(sampleRate);
        _DSites.push_back(latitude);
        _DSites.push_back(longitude);
        SitesList = SitesList->NextSiblingElement("Site");
    }
}

void MemManager::importKpData(){
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError eResult;
    _DKpIndex.clear();
    _DKpIndex.shrink_to_fit();
    std::string KpStr = this->_testDirectory;
    KpStr.append("/Kp.xml");
    eResult = doc.LoadFile(KpStr.c_str());
    XMLCheckResult(eResult);

    tinyxml2::XMLNode *pRoot = doc.FirstChild();
    if(pRoot == NULL){
        std::cout<<"file read error"<<std::endl;
        exit(tinyxml2::XML_ERROR_FILE_READ_ERROR);
    }
    tinyxml2::XMLElement * pElement = pRoot->NextSiblingElement("Kp");
    if(pElement == NULL){
        std::cout<<"error parsing element"<<std::endl;
        exit(tinyxml2::XML_ERROR_PARSING_ELEMENT);
    }
    tinyxml2::XMLElement * KpList = pElement->FirstChildElement("Kp_hr");
    while(KpList != NULL){
        int seconds;
        float magnitude;
        eResult = KpList->QueryIntAttribute("secs", &seconds);
        XMLCheckResult(eResult);
        eResult = KpList->QueryFloatText(&magnitude);
        XMLCheckResult(eResult);
        _DKpIndex.push_back(seconds);
        _DKpIndex.push_back(magnitude);
        KpList = KpList->NextSiblingElement("Kp_hr");
    }
}

void MemManager::importGQuakes(){
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError eResult;
    _DGQuakes.clear();
    _DGQuakes.shrink_to_fit();
    std::string QuakesStr = this->_testDirectory;
    QuakesStr.append("/Quakes.xml");
    eResult = doc.LoadFile(QuakesStr.c_str());
    XMLCheckResult(eResult);
    tinyxml2::XMLNode *pRoot = doc.FirstChild();
    if(pRoot == NULL){
        std::cout<<"file read error"<<std::endl;
        exit(tinyxml2::XML_ERROR_FILE_READ_ERROR);
    }
    tinyxml2::XMLElement * pElement = pRoot->NextSiblingElement("Quakes");
    if(pElement == NULL){
        std::cout<<"error parsing element"<<std::endl;
        exit(tinyxml2::XML_ERROR_PARSING_ELEMENT);
    }
    tinyxml2::XMLElement * quakeList = pElement->FirstChildElement("Quake");
    while(quakeList != NULL){
        int seconds;
        float latitude, longitude, magnitude, depth;
        eResult = quakeList->QueryIntAttribute("secs", &seconds);
        XMLCheckResult(eResult);
        eResult = quakeList->QueryFloatAttribute("latitude", &latitude);
        XMLCheckResult(eResult);
        eResult = quakeList->QueryFloatAttribute("longitude", &longitude);
        XMLCheckResult(eResult);
        eResult = quakeList->QueryFloatAttribute("magnitude", &magnitude);
        XMLCheckResult(eResult);
        eResult = quakeList->QueryFloatAttribute("depth", &depth);
        _DGQuakes.push_back(seconds);
        _DGQuakes.push_back(latitude);
        _DGQuakes.push_back(longitude);
        _DGQuakes.push_back(magnitude);
        _DGQuakes.push_back(depth);
        quakeList = quakeList->NextSiblingElement("Quake");
    }
}

void MemManager::importTrainingData(){ // this is only called once for the entire life of the program, also uses CSV so it's done with fopen
    std::string answerStr = this->_dataDirectory;
    answerStr.append("/gtf.csv");
    std::ifstream answerfile(answerStr.c_str());
    std::string line;
    std::getline(answerfile, line);
    int numOfTests;
    std::istringstream(line) >> numOfTests;
    _DTraining.push_back(numOfTests);
    while(std::getline(answerfile, line)){
        std::vector<std::string> token;
        std::string item;
        std::stringstream ss(line);
        while(std::getline(ss,  item, ',')){
            token.push_back(item);
        }
        _DTraining.push_back(std::atoi(token[0].c_str())); // setID
        std::string startTime = token[1];
        std::string EqTime = token[2];
        _DTraining.push_back(timeDifferenceCalculation(startTime, EqTime));
        _DTraining.push_back(std::atoi(token[3].c_str())); // magnitude
        _DTraining.push_back(std::atoi(token[4].c_str())); // latitude
        _DTraining.push_back(std::atoi(token[5].c_str())); // longitude
        _DTraining.push_back(std::atoi(token[6].c_str())); // siteNumber
        _DTraining.push_back(std::atoi(token[7].c_str())); // distance to quake
    }
}
