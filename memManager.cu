#include "memManager.h"

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do{cudaError_t err = call; if (cudaSuccess != err) {fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",__FILE__, __LINE__, cudaGetErrorString(err) ); exit(EXIT_FAILURE);}} while (0)
#endif
void* memManager::alloc(size_t len){
    void *ptr;
    cudaMallocManaged(&ptr, len);
    return ptr;

}

void memManager::dealloc(void* ptr){
    cudaDeviceSynchronize();
    cudaFree(ptr);
}






//bool MemManager::memoryAlloc(int individualLength,float pMax){//allocates memory for genetis & input vectors
//    long long hostMem = GetHostRamInBytes()*pMaxHost; //make a host memory container, this is the max
//    long long deviceMem = GetDeviceRamInBytes()*pMaxDevice; //dito for gpu
//    _hostGeneticsAlloc = hostMem/8; //since these are doubles, divide bytes by 8
//    _deviceGeneticsAlloc = deviceMem/8;
//    _hostGeneticsAlloc = (_hostGeneticsAlloc/individualLength)*individualLength;
//    _deviceGeneticsAlloc = (_deviceGeneticsAlloc/individualLength)*individualLength;
//    std::cerr<<"about to allocate: "<<_hostGeneticsAlloc*8<<" bytes for the host"<<std::endl;
//    std::cerr<<"about to allocate: "<<_deviceGeneticsAlloc*8<<" byes for the device"<<std::endl;
//    //initialize all large vectors (everything not from an xml file)
//    try{
//        this->_HGenetics = new thrust::host_vector<double>(_hostGeneticsAlloc);
//    }
//    catch(thrust::system_error &e){
//        std::cerr<<"Error resizing vector Element: "<<e.what()<<std::endl;
//        exit(1);
//    }
//    catch(std::bad_alloc &e){
//        std::cerr<<"Ran out of space due to : "<<"host"<<std::endl;
//        std::cerr<<e.what()<<std::endl;
//        std::cout<<GetHostRamInBytes()<<std::endl;
//        exit(1);
//    }
//    try{
//        this->_DGenetics = new thrust::device_vector<double>(_deviceGeneticsAlloc);

//    }
//    catch(thrust::system_error &e){
//        std::cerr<<"Error resizing vector Element: "<<e.what()<<std::endl;
//        exit(1);
//    }
//    catch(std::bad_alloc &e){
//        std::cerr<<"Ran out of space due to : "<<"device"<<std::endl;
//        std::cerr<<e.what()<<std::endl;
//        std::cout<<GetDeviceRamInBytes()<<std::endl;
//        exit(1);
//    }
//    std::cerr<<"gpu ram avilable after genetics allocation: "<<GetDeviceRamInBytes()<<std::endl;
//    std::cerr<<"host ram available: "<<GetHostRamInBytes()<<std::endl;

//    return true;
//}

//bool MemManager::GeneticsPushToHost(){
//    long long dGenLength = _DGenetics->size();
//    long long currpos = dGenLength + genItr;
//    if(currpos*2 <= _HGenetics->size() && currpos < _HGenetics->size()){ //if _HGenetics can take 2 more at the current size, keep going
//        thrust::copy(_DGenetics->begin(), _DGenetics->end(), _HGenetics->begin()+genItr);
//        genItr = currpos; //set the iterator  to the new position.
//        return true;
//    }
//    else if(currpos*2 > _HGenetics->size() && currpos < _HGenetics->size()){//if _HGenetics can only take 1 or exactly 2 at current size, resize dgen to fit
//        thrust::copy(_DGenetics->begin(), _DGenetics->end(), _HGenetics->begin()+genItr);
//        _DGenetics.resize(_HGenetics->size() - currpos);// the device_vector for genetics was resized to fit the remaining host mem container.
//        genItr = currpos;
//        return true;
//    }
//    else if(currpos == _HGenetics->size() && dGenLength !=0){//if the _HGenetics vector is full, tell the GPU
//        thrust::copy(dGen->_array, dGen->_array + dGenLength, _HGenetics->begin()+genItr);
//        genItr = 0;
//        _DGenetics->resize(_deviceGeneticsAlloc);
//        return false;
//    }
//    else if(currpos > _HGenetics->size()){
//        std::cout<<"#4"<<std::endl;
//        return false;
//    }
//    return false;
//}


