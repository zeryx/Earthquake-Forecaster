#include <prep.h>
#include <connections.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/error/error.h>
#include <rapidjson/error/en.h>
#include <cstdio>
#include <getsys.h>
#include <iostream>
#include <fstream>

prep::prep(){

}
prep::~prep(){

}

bool prep::checkForGenomes(const char* filepath){
    FILE* gFile = std::fopen(filepath, "r");
    std::cerr<<"checking for genomes file.."<<std::endl;
    bool chk = false;
    if(gFile){
        std::cerr<<"the genomes Files exists"<<std::endl;
        fclose(gFile);
        chk = true;
    }
    else{
        std::cerr<<"no genomes file found"<<std::endl;
        chk= false;
    }
    return chk;
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
    _siteData = siteData;
    _istraining = false;
    _net.confTestParams(SiteNum, sampleRate);
    return true;
}

bool prep::readNetParmeters(const char *filepath){
    rapidjson::Document doc;
    FILE* netParamsFile = fopen(filepath, "r");
    char buffer[65536];
    rapidjson::FileReadStream in(netParamsFile, buffer, sizeof(buffer));
    if(doc.ParseStream(in).HasParseError()){
        std::cerr<<"netParams has parse error"<<std::endl;
        return false;
    }
    if(!doc.IsObject()){
        std::cerr<<"not in json format or file doesn't exist."<<std::endl;
        return false;
    }
    if(!doc.HasMember("neurons")){
        std::cerr<<"no member named neurons exists in file"<<std::endl;
        return false;
    }
    rapidjson::Value &a = doc["neurons"];
    int input, hidden, smemory, lmemory, memGateIn, memGateOut, memGateForget, output;
    for(rapidjson::Value::ConstMemberIterator itr = a.MemberBegin();
        itr != a.MemberEnd(); ++itr){
        std::string tmp = itr->name.GetString();
        if(tmp == "input")
            input = itr->value.GetInt();

        else if(tmp == "hidden")
            hidden = itr->value.GetInt();

        else if(tmp == "smemory")
            smemory = itr->value.GetInt();

        else if(tmp == "lmemory")
            lmemory = itr->value.GetInt();

        else if(tmp == "memIn")
            memGateIn = itr->value.GetInt();

        else if(tmp == "memOut")
            memGateOut = itr->value.GetInt();

        else if(tmp == "memForget")
            memGateForget = itr->value.GetInt();

        else if(tmp == "output")
            output = itr->value.GetInt();

        else{
            std::cerr<<"invalid neuron type"<<std::endl;
            return false;
        }
    }
    std::fclose(netParamsFile);
    _net.confNetParams(input, hidden, memory, memGateIn, memGateOut, memGateForget, output);
    return true;
}

bool prep::readOrders(const char* filepath){
    rapidjson::Document doc;
    FILE* orderFile = fopen(filepath, "r");
    char buffer[65536];
    rapidjson::FileReadStream in(orderFile, buffer, sizeof(buffer));
    if(doc.ParseStream(in).HasParseError()){
        std::cerr<<"orders has parse error"<<std::endl;
        return false;

    }
    assert(doc.IsObject());
    rapidjson::Value &orders = doc["orders"];
    assert(orders.IsArray());
    int weights =0;
    int size = orders.Size();
    int numOrders = size;
    _connections = new Order[size];
    for(size_t itr=0; itr<orders.Size(); itr++){
        Order tmp;
        std::string def1 = orders[itr]["first"]["def"].GetString();
        std::string def2 = orders[itr]["second"]["def"].GetString();
        tmp.first.id = orders[itr]["first"]["id"].GetInt();
        tmp.second.id = orders[itr]["second"]["id"].GetInt();
        tmp.first.def = this->enumStringcmp(def1);
        tmp.second.def= this->enumStringcmp(def2);
        if(orders[itr].HasMember("third")){
            std::string def3 = orders[itr]["third"]["def"].GetString();
            tmp.third.id = orders[itr]["third"]["id"].GetInt();
            tmp.third.def = this->enumStringcmp(def3);
            if(tmp.third.def == typeNULL){
                std::cerr<<"invalid descriptor for third parameter, at number"<<itr<<std::endl;
                return false;
            }
        }
        else{
            tmp.third.def = typeNULL;
            tmp.third.id =0;
        }
        if(tmp.first.def == typeNULL || tmp.second.def == typeNULL){
            std::cerr<<"invalid descriptor for first &/or second parameter, at number"<<itr<<std::endl;
            return false;
        }


        //check if weights should be incremented.
        if(tmp.second.def != typeZero
                && tmp.second.def != typeSquash
                && tmp.third.def == typeNULL)
            weights++;
        _connections[itr] = tmp;
    }
    std::cerr<<"number of weights: "<<weights<<std::endl;
    _net.confOrder( numOrders, weights);
    fclose(orderFile);
    return true;
}

neuroType prep::enumStringcmp(std::string def){

    neuroType ret = typeNULL;

    if(def == "input")
        ret = typeInput;

    else if(def == "hidden")
        ret = typeHidden;

    else if(def == "smemory")
        ret = typeShortMemory;

    else if(def == "lmemory")
        ret = typeLongMemory;

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

void prep::EndOfTrial(const char* filepath){
    _net.endOfTrial();
    std::ofstream ret;
    ret.open(filepath,  std::ofstream::trunc | std::ifstream::binary);
    _net.saveToFile(ret);
    ret.close();
    std::cerr<<"weights stored successfully"<<std::endl;

}


void prep::hotStart(const char* filepath){
    std::ifstream gstream;
    gstream.open(filepath, std::ifstream::ate | std::ifstream::binary);
    _net.loadFromFile(gstream);
    gstream.close();
}

void prep::coldStart(){
    _net.allocateHostAndGPUObjects(GetDeviceRamInBytes()*0.85, GetHostRamInBytes()*0.40);
    _net.generateWeights();

}

void prep::forecast(std::vector<double> &ret, int &hour, std::vector<int> &data, double &K, std::vector<double> &globalQuakes){
    if(this->_istraining)
        _net.trainForecast(&ret, hour, data, K, globalQuakes, _connections, _answers, *_siteData);
    else
        _net.challengeForecast(&ret, hour, data, K, globalQuakes, _connections, *_siteData);
}
