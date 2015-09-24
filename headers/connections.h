#ifndef ORDER_H
#define ORDER_H
#include <cuda_runtime.h>
enum neuroNouns{
    nounNULL =0,
    nounInput =1,
    nounHidden =2,
    nounMemory = 3,
    nounMemGateIn = 4,
    nounMemGateOut = 5,
    nounMemGateForget = 6,
    nounOutput = 7,
    nounBias =10
};

enum neuroVerbs{
    verbNULL =0,
    verbZero =1,
    verbSquash =2,
    verbPush = 3,
    verbMemGate = 4,
};

struct Noun{
    neuroNouns def;
    int id;
};

struct Verb{
    neuroVerbs def;
};

class Order{ // order contains any number of nouns between 1-4, and a single verb

public:
    Order();
    Order(Noun first, Verb verb);
    Order(Noun first, Noun second, Verb verb);
    Order(Noun first, Noun second, Noun third, Verb verb);
    Order(Noun first, Noun second, Noun third, Noun fourth, Verb verb);

     __host__ __device__ Noun first();
     __host__ __device__ Noun second();
     __host__ __device__ Noun third();
     __host__ __device__ Noun fourth();
     __host__ __device__ Verb verb();

    __host__ __device__ void setFirst(neuroNouns def, int id);
    __host__ __device__ void setSecond(neuroNouns def, int id);
    __host__ __device__ void setThird(neuroNouns def, int id);
    __host__ __device__ void setFourth(neuroNouns def, int id);
   __host__ __device__  void setVerb(neuroVerbs def);
    ~Order();

public:
    Noun _first;
    Noun _second;
    Noun _third;
    Noun _fourth;
    Verb _verb;
};


#endif
