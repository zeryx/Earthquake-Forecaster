#ifndef CONNECTIONS_H
#define CONNECTIONS_H
#include <string>
enum neuroNoun{
    typeNULL =0,
    nounInput =1,
    nounHidden =2,
    nounShortMemory = 3,
    nounLongMemory = 4,
    nounMemGateIn = 5,
    nounMemGateOut = 6,
    nounMemGateForget = 7,
    nounOutput = 8,
    nounBias =9
};

enum neuroVerb{
    verbSquash =10,
    verbZero =11,
    verbPush = 12,
    verbMemGate = 13,
    verbMemForget = 14
};

struct Noun{
    neuroNoun def;
    H id;
};
struct Verb{
    neuroVerb def;
};

struct Order{
    Noun first;
    Noun second;
    Noun third;
//    noun fourth; //unused
    Verb verb;
};
