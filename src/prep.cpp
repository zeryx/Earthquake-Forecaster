#include <prep.h>
#include <connections.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/error/error.h>
#include <rapidjson/error/en.h>

#include <getsys.h>
#include <iostream>
#include <fstream>

prep::prep(){

}
prep::~prep(){

}

void prep::storeGenomes(const char* filepath){
    //    std::ofstream ret;
    //    ret.open(filepath.c_str(), std::ios_base::out | std::ios_base::trunc);
    //    size_t host_offset=0;
    //    for(int i=0; i<_net._numOfStreams; i++){
    //        for(int k=0; k<_net._streamSize; k++){
    //            ret << _net.device_genetics.array[i+host_offset]<<",";
    //        }
    //        ret<<std::endl;
    //        host_offset += _net._streamSize;
    //    }
    //    ret.close();

}

bool prep::checkForGenomes(const char* filepath){
    FILE* gFile = std::fopen(filepath, "r");
    std::cerr<<"checking for genomes file.."<<std::endl;
    if(gFile){
        std::cerr<<"the genomes Files exists"<<std::endl;
        std::fclose(gFile);
        return true;
    }
    else{
        std::cerr<<"no genomes file found"<<std::endl;
        std::fclose(gFile);
        return false;
    }
}

void prep::doingTraining(int site, int hour, double lat,
                         double lon, double mag, double dist){
    _answers.push_back(site);
    _answers.push_back(hour);
    _answers.push_back(lat);
    _answers.push_back(lon);
    _answers.push_back(mag);
    _answers.push_back(dist);
    _istraining = true;

}

bool prep::init(int sampleRate, int SiteNum, std::vector<double> *siteData){
    _net.setParams(22, sampleRate);
    _net.setParams(23, SiteNum);
    _siteData = siteData;
    _istraining = false;
    _net.setParams(2, _net._hostParams.array[0] + _net._hostParams.array[1] + 2 + 3*SiteNum); //1*numOfSites for community mag, 1 for fitness, and 1 for age.
    return true;
}

bool prep::checkForJson(const char* filepath){
    rapidjson::Document doc;
    FILE* orderFile = fopen(filepath, "r");
    char buffer[65536];
    rapidjson::FileReadStream in(orderFile, buffer, sizeof(buffer));
    if(doc.ParseStream(in).HasParseError()){

        std::cerr<<"parse error"<<std::endl;
        exit(1);

    }
    assert(doc.IsObject());
    assert(doc.HasMember("neurons"));
    rapidjson::Value &a = doc["neurons"];
    int input, hidden, memory, memGateIn, memGateOut, memGateForget, output, weights=0;
    for(rapidjson::Value::ConstMemberIterator itr = a.MemberBegin();
        itr != a.MemberEnd(); ++itr){
        std::string tmp = itr->name.GetString();
        if(tmp == "input")
            input = itr->value.GetInt();

        else if(tmp == "hidden")
            hidden = itr->value.GetInt();

        else if(tmp == "memory")
            memory = itr->value.GetInt();

        else if(tmp == "memIn")
            memGateIn = itr->value.GetInt();

        else if(tmp == "memOut")
            memGateOut = itr->value.GetInt();

        else if(tmp == "memForget")
            memGateForget = itr->value.GetInt();

        else if(tmp == "output")
            output = itr->value.GetInt();
    }
    rapidjson::Value &orders = doc["orders"];
    assert(orders.IsArray());
    _connections = new Order[orders.Size()];
    _net.setParams(26, orders.Size());
    for(size_t itr=0; itr<orders.Size(); itr++){
        Order tmp;
        std::string def1 = orders[itr]["first"]["def"].GetString();
        std::string def2 = orders[itr]["second"]["def"].GetString();
        tmp.first.id = orders[itr]["first"]["id"].GetInt();
        tmp.second.id = orders[itr]["second"]["id"].GetInt();
        tmp.first.def = this->strcmp(def1);
        tmp.second.def= this->strcmp(def2);
        if(orders[itr].HasMember("third")){
            std::string def3 = orders[itr]["third"]["def"].GetString();
            tmp.third.id = orders[itr]["third"]["id"].GetInt();
            tmp.third.def = this->strcmp(def3);
            if(tmp.third.def == typeNULL){
                std::cerr<<"invalid descriptor for third parameter, at number"<<itr<<std::endl;
                exit(1);
            }
        }
        else{
            tmp.third.def= typeNULL;
            tmp.third.id =0;
        }
        if(tmp.first.def == typeNULL || tmp.second.def == typeNULL){
            std::cerr<<"invalid descriptor for first &/or second parameter, at number"<<itr<<std::endl;
            exit(1);
        }

        //check if weights should be incremented.
        if(tmp.first.def != typeMemory
                && tmp.second.def != typeMemory
                && tmp.second.def != typeZero
                && tmp.second.def != typeSquash
                && tmp.third.def != typeNULL)
            weights++;
        _connections[itr] = tmp;
    }
    _net.confBasicParams(input, hidden, memory, memGateIn, memGateOut, memGateForget, output, weights);
    fclose(orderFile);
    return true;
}

neuroType prep::strcmp(std::string def){

    neuroType ret = typeNULL;

    if(def == "input")
        ret = typeInput;

    else if(def == "hidden")
        ret = typeHidden;

    else if(def == "memory")
        ret = typeMemory;

    else if(def == "memGateIn")
        ret = typeMemGateIn;

    else if(def == "memGateOut")
        ret = typeMemGateOut;

    else if(def == "memGateForget")
        ret = typeMemGateForget;

    else if(def == "output")
        ret = typeOutput;

    else if(def == "zero")
        ret = typeZero;

    else if(def == "bias")
        ret = typeBias;

    else if(def == "squash")
        ret = typeSquash;

    return ret;
}
void prep::hotStart(std::string filename, float pmax){
    std::ifstream gstream;
    gstream.open(filename.c_str(), std::ios_base::binary | std::ios_base::ate);
    _net.loadFromFile(gstream, pmax);
    _net.generateWeights();
}

void prep::coldStart(float pmax){
    _net.allocateHostAndGPUObjects(pmax, GetDeviceRamInBytes(), GetHostRamInBytes());
}

void prep::forecast(std::vector<double> &ret, int &hour, std::vector<int> &data, double &K, std::vector<double> &globalQuakes){
    if(this->_istraining)
        _net.trainForecast(&ret, hour, data, K, globalQuakes, _connections, _answers, *_siteData);
    else
        _net.challengeForecast(&ret, hour, data, K, globalQuakes, _connections, *_siteData);
}