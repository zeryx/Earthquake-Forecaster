#include "memManager.h"
#include "getsys.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/swap.h>
#include <fstream>
#include <sstream>
#include <ostream>
#include <thrust/system_error.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>


MemManager::MemManager(){}


dataArray<double> MemManager::genetics(){
    return convertToKernel(_DGenetics);
}




bool MemManager::memoryAlloc(int individualLength, float pMaxHost, float pMaxDevice){//allocates memory for genetis & input vectors
    //    long long hostMem = GetHostRamInBytes()*pMaxHost; //make a host memory container, this is the max
    long long deviceMem = GetDeviceRamInBytes()*pMaxDevice; //dito for gpu
    //    _hostGeneticsAlloc = hostMem/8; //since these are doubles, divide bytes by 8
    _deviceGeneticsAlloc = deviceMem/8;

    //    _hostGeneticsAlloc = (_hostGeneticsAlloc/individualLength)*individualLength;
    _deviceGeneticsAlloc = (_deviceGeneticsAlloc/individualLength)*individualLength;
    //initialize all large vectors (everything not from an xml file)
    try{
        //        this->_HGenetics.setMax(_hostGeneticsAlloc);
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
//bool MemManager::geneticsBufferSwap(dataArray<double> *dGen){
//    return false;
//}

//bool MemManager::GeneticsPushToHost(dataArray<double> *dGen){
//    long long dGenLength = dGen->_size;
//    long long currpos = dGenLength + _HGenetics._itr;
//    std::cout<<"length of device vector: "<<dGenLength<<std::endl;
//    std::cout<<"host vector max length: "<<_HGenetics._maxLen<<std::endl;
//    if(currpos*2 <= _HGenetics._maxLen && currpos < _HGenetics._maxLen){ //if _HGenetics can take 2 more at the current size, keep going
//        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics._hVect.begin()+_HGenetics._itr);
//        _HGenetics._itr = currpos; //set the iterator  to the new position.
//        std::cout<<"#1"<<std::endl;
//        return true;
//    }
//    else if(currpos*2 > _HGenetics._maxLen && currpos < _HGenetics._maxLen){//if _HGenetics can only take 1 or exactly 2 at current size, resize dgen to fit
//        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics._hVect.begin()+_HGenetics._itr);
//        _DGenetics.resize(_HGenetics._maxLen - currpos);// the device_vector for genetics was resized to fit the remaining host mem container.
//        std::cout<<"resized genetics to: "<<_HGenetics._maxLen - _HGenetics._itr<<std::endl;
//        dGen->_size = _DGenetics.size();
//        _HGenetics._itr = currpos;
//        std::cout<<"#2"<<std::endl;
//        return true;
//    }
//    else if(currpos == _HGenetics._maxLen && dGenLength !=0){//if the _HGenetics vector is full, tell the GPU
//        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics._hVect.begin()+_HGenetics._itr);
//        _HGenetics._itr = 0;
//        _DGenetics.resize(_deviceGeneticsAlloc);
//        dGen->_size =_DGenetics.size();
//        std::cout<<"#3"<<std::endl;
//        return false;
//    }
//    else if(currpos > _HGenetics._maxLen){
//        std::cout<<"#4"<<std::endl;
//        return false;
//    }
//    return false;
//}

void MemManager::initFromStream(std::ifstream &stream){
    std::string line;
    int itr =0;
    while(std::getline(stream, line)){ // each line
        std::string item;
        std::stringstream ss(line);
        while(std::getline(ss, item, ',')){ // each weight
            _DGenetics[itr] = std::atoi(item.c_str());
        }
    }
}

void MemManager::pushToStream(std::string filename){
    std::ofstream ret;
    ret.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
    for(int i=0; i<_DGenetics.size(); i++){
        ret << _DGenetics[i]<<","<<std::endl;
    }
    ret.close();
}

