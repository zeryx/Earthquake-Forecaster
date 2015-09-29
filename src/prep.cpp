#include <prep.h>
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

bool prep::checkForGenomes(){
    std::string location= "wt" +_trainingNum + ".bin";
    FILE* gFile = std::fopen(location.c_str(), "r");
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
    if(!doc.HasMember("neurons") || !doc.HasMember("trainingSet")){
        std::cerr<<"net.json is missing parameters"<<std::endl;
        return false;
    }
    rapidjson::Value &set = doc["trainingSet"];
    _trainingNum = set.GetString();
    std::cerr<<"training set number is: "<<_trainingNum<<std::endl;
    rapidjson::Value &a = doc["neurons"];
    int input, hidden, memory, memGateIn, memGateOut, memGateForget, output;
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
        std::string verb1 = orders[itr]["verb"]["def"].GetString();

        tmp.setFirst(this->nounStringcmp(def1), orders[itr]["first"]["id"].GetInt());
        tmp.setVerb(this->verbStringcmp(verb1));

        if(orders[itr].HasMember("second")){
                    std::string def2 = orders[itr]["second"]["def"].GetString();

                    tmp.setSecond(this->nounStringcmp(def2), orders[itr]["second"]["id"].GetInt());
        }
        else{
            tmp.setSecond(nounNULL, 0);
        }

        if(orders[itr].HasMember("third")){
            std::string def2 = orders[itr]["third"]["def"].GetString();

            tmp.setThird(this->nounStringcmp(def2), orders[itr]["third"]["id"].GetInt());
            if(tmp.third().def == nounNULL){
                std::cerr<<"invalid descriptor for third parameter, at number"<<itr<<std::endl;
                return false;
            }
        }
        else{
            tmp.setThird(nounNULL, 0);
        }

        if(orders[itr].HasMember("fourth")){
            std::string def2 = orders[itr]["fourth"]["def"].GetString();
            tmp.setFourth(this->nounStringcmp(def2), orders[itr]["fourth"]["id"].GetInt());

            if(tmp.fourth().def == nounNULL){
                std::cerr<<"invalid descriptor for third parameter, at number"<<itr<<std::endl;
                return false;
            }
        }
        else{
            tmp.setFourth(nounNULL, 0);
        }

        if(tmp.first().def == nounNULL || tmp.verb().def == verbNULL){
            std::cerr<<"invalid descriptor for first &/or second parameter, at number"<<itr<<std::endl;
            return false;
        }


        //check if weights should be incremented.
        if(tmp.verb().def != verbZero
                && tmp.verb().def != verbSquash
                && tmp.third().def == nounNULL)
            weights++;

        _connections[itr] = tmp;
    }
    std::cerr<<"number of weights: "<<weights<<std::endl;
    _net.confOrder( numOrders, weights);
    fclose(orderFile);
    return true;
}

neuroNouns prep::nounStringcmp(std::string def){

    neuroNouns ret = nounNULL;

    if(def == "input")
        ret = nounInput;

    else if(def == "hidden")
        ret = nounHidden;

    else if(def == "memory")
        ret = nounMemory;

    else if(def == "memGateIn")
        ret = nounMemGateIn;

    else if(def == "memGateOut")
        ret = nounMemGateOut;

    else if(def == "memGateForget")
        ret = nounMemGateForget;

    else if(def == "output")
        ret = nounOutput;

    else if(def == "bias")
        ret = nounBias;


    return ret;
}

neuroVerbs prep::verbStringcmp(std::string def){

    neuroVerbs ret = verbNULL;

    if(def == "squash")
        ret = verbSquash;

    else if(def == "zero")
        ret = verbZero;

    else if(def == "push")
        ret = verbPush;

    else if(def == "memGate")
        ret = verbMemGate;

    return ret;
}

void prep::EndOfTrial(){
    _net.training();
    std::ofstream ret;
    std::string location= "wt" +_trainingNum + ".bin";
    ret.open(location.c_str(),  std::ofstream::trunc | std::ifstream::binary);
    _net.saveToFile(ret);
    ret.close();
    std::cerr<<"weights stored successfully"<<std::endl;
}


void prep::hotStart(){
    std::ifstream gstream;
    std::string location= "wt" +_trainingNum + ".bin";
    gstream.open(location.c_str(), std::ifstream::ate | std::ifstream::binary);
    _net.loadFromFile(gstream);
    gstream.close();
}

void prep::coldStart(){
    _net.allocateHostAndGPUObjects(GetDeviceRamInBytes()*0.75, GetHostRamInBytes()*0.2);
    _net.generateWeights();

}

void prep::forecast(std::vector<double> &ret, int &hour, std::vector<int> &data, double &K, std::vector<double> &globalQuakes){
    if(this->_istraining)
        _net.trainForecast(&ret, hour, data, K, globalQuakes, _connections, _answers, *_siteData);
    else
        _net.challengeForecast(&ret, hour, data, K, globalQuakes, _connections, *_siteData);
}
