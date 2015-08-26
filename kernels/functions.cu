#include <kernelDefs.h>

__host__ __device__ float bearingCalc(float lat1, float lon1, float lat2, float lon2){

    float y = sin(lon2-lon1) * cos(lat2);
    float x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2-lon1);

    float brng = atan2(y, x);

    brng = brng*180/M_PI;
    brng += 180;
    while(brng>=360)
        brng -= 360;
    return brng;
}

__host__ __device__ float distCalc(float lat1, float lon1, float lat2, float lon2){
    const float earthRad = 6371.01;
    float dLon = (lon1 - lon2);
    float dlat = (lat1 - lat2);
    lat1 = lat1;
    lat2 = lat2;
    float x = sin(dlat/2) * sin(dlat/2) + cos(lat1) * cos(lat2) * sin(dLon/2) * sin(dLon/2);
    float c = 2*atan2(sqrt(x), sqrt(1-x));

    return earthRad*c;
}

__host__ __device__ float normalize(float x, float mean, float stdev){
    return (fabs(x-mean))/(stdev*2);
}

__host__ __device__ float shift(float x, float max, float min){
    return (x-min)/(max-min);
}

__host__ __device__ float ActFunc(float x){
    return tanh(x);
}

__host__ __device__ double scoreFunc(double whenGuess, double whenAns, int hour, float latGuess, float lonGuess, float latAns, float lonAns){
    return 1/(fabs(whenGuess-whenAns-hour)*distCalc(latGuess, lonGuess, latAns, lonAns));
}

