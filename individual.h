#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H
#include <thrust/device_vector.h>

class Individual{//stores the weights and its fitness values
public:
    Individual();
    std::vector<double> _weights;
    bool setAbsFitness(double newAbsFitness);
    double absFitness();//gets the absolute fitness value
    bool calcRelativeFitness(double averageAbsFitness);//calculate the relative fitness
    double relativeFitness();//gets the normalized fitness value from all individuals
private:
    double _absoluteFitness, _relativeFitness;

};


#endif
