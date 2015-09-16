#include <connections.h>

Order::Order(){
    this->setFirst(nounNULL, 0);
    this->setSecond(nounNULL, 0);
    this->setThird(nounNULL, 0);
    this->setFourth(nounNULL, 0);
    this->setVerb(verbNULL);
}

Order::Order(Noun first, Verb verb){
    this->setFirst(first.def, first.id);
    this->setSecond(nounNULL, 0);
    this->setThird(nounNULL, 0);
    this->setFourth(nounNULL, 0);
    this->setVerb(verb.def);
}

Order::Order(Noun first, Noun second, Verb verb){
    this->setFirst(first.def, first.id);
    this->setSecond(second.def, second.id);
    this->setThird(nounNULL, 0);
    this->setFourth(nounNULL, 0);
    this->setVerb(verb.def);
}

Order::Order(Noun first, Noun second, Noun third, Verb verb){
    this->setFirst(first.def, first.id);
    this->setSecond(second.def, second.id);
    this->setThird(third.def, third.id);
    this->setFourth(nounNULL, 0);
    this->setVerb(verb.def);
}


Order::Order(Noun first, Noun second, Noun third, Noun fourth, Verb verb){
    this->setFirst(first.def, first.id);
    this->setSecond(second.def, second.id);
    this->setThird(third.def, third.id);
    this->setFourth(fourth.def, fourth.id);
    this->setVerb(verb.def);
}


  __host__ __device__ Noun Order::first(){
    return this->_first;
}

  __host__ __device__ Noun Order::second(){
    return this->_second;
}

  __host__ __device__ Noun Order::third(){
    return this->_third;
}

  __host__ __device__ Noun Order::fourth(){
    return this->_fourth;
}

  __host__ __device__ Verb Order::verb(){
    return this->_verb;
}

__host__ __device__ void Order::setFirst(neuroNouns def, int id){
    this->_first.def = def;
    this->_first.id = id;
}

__host__ __device__ void Order::setSecond(neuroNouns def, int id){
    this->_second.def = def;
    this->_second.id = id;
}

__host__ __device__ void Order::setThird(neuroNouns def, int id){
    this->_third.def = def;
    this->_third.id = id;
}

__host__ __device__ void Order::setFourth(neuroNouns def, int id){
    this->_fourth.def = def;
    this->_fourth.id = id;
}

__host__ __device__ void Order::setVerb(neuroVerbs def){
    this->_verb.def = def;
}

Order::~Order(){}
