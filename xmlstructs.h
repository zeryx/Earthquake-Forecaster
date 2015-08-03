#ifndef XMLSTRUCTS_H
#define XMLSTRUCTS_H
#include <ctime>
//small little structs for working with the XML documents easier (rather than converting all to one data type, keep them as their components
struct Kp{
    long seconds;
    float magnitude;
};

struct GQuakes{
    long seconds;
    float longitude;
    float latitude;
    float depth;
    float magnitude;
};

struct SiteInfo{
    int sampleRate;
    int siteNumber;
    float longitude;
    float latitude;
};

struct Answers{
    int setID;
    int siteNum;
    int hrOfQuake;
    float latitude;
    float longitude;
    float magnitude;
    float distToQuake;
};

#endif
