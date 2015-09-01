#ifndef PREP_H
#define PREP_H
#include <network.h>
#include <string>

class prep{
public:
    prep();

    ~prep();

    void hotStart(std::string filename, float pmax);

    void coldStart(float pmax);

    void doingTraining(int site, int hour, double lat,
                       double lon, double mag, double dist);
    bool checkForJson(const char* filepath);

    bool checkForGenomes(const char* filepath);

    void storeGenomes(const char* filepath);

    bool init(int sampleRate, int SiteNum, std::vector<double> *siteData);

    void forecast(std::vector<double> &ret, int &hour, std::vector<int> &data, double &K, std::vector<double> &globalQuakes);

    neuroType strcmp(std::string);
private:
    NetworkGenetic _net;
    Order *_connections;
    std::vector<double> _answers;
    std::vector<double> *_siteData;
    bool _istraining;


};

#endif