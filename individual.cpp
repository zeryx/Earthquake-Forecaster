#include "individual.h"

Individual::Individual(){

}


//sets

bool Individual::setAbsFitness(double newAbsFitness){
    if(newAbsFitness != _absoluteFitness && newAbsFitness >=0){ //fitness is always greater or equal to zero
        _absoluteFitness = newAbsFitness;
        return true;
    }
    else
        return false;
}

bool Individual::calcRelativeFitness(double averageAbsFitness){
    if(averageAbsFitness >0){ // this value should never be equal to zero unless something broke.
        _relativeFitness = _absoluteFitness/averageAbsFitness; // if greater than 1, this individual is better than the average.
        return true;
    }
    return false;
}


//gets
double Individual::absFitness(){
    return _absoluteFitness;
}

double Individual::relativeFitness(){
    return _relativeFitness;
}
