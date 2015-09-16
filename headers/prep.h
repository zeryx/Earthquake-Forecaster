#ifndef PREP_H
#define PREP_H
#include <network.h>
#include <string>
#include <connections.h>


class prep{
public:
    prep();

    ~prep();

    void hotStart(const char*);

    void coldStart();

    void doingTraining(int site, int hour, double lat,
                       double lon, double mag, double dist);

    bool readNetParmeters(const char* filepath);

    bool readOrders(const char* filepath);

    bool checkForGenomes(const char* filepath);

    void EndOfTrial(const char* filepath);

    bool init(int sampleRate, int SiteNum, std::vector<double> *siteData);

    void forecast(std::vector<double> &ret, int &hour, std::vector<int> &data, double &K, std::vector<double> &globalQuakes);

    neuroNouns nounStringcmp(std::string);

    neuroVerbs verbStringcmp(std::string);

private:
    NetworkGenetic _net;
    Order *_connections;
    std::vector<double> _answers;
    std::vector<double> *_siteData;
    bool _istraining;


};

#endif
