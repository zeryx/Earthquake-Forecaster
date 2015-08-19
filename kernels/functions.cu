#include <kernelDefs.h>

__host__ __device__ double bearingCalc(double lat1, double lon1, double lat2, double lon2){

    double y = sin(lon2-lon1) * cos(lat2);
    double x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2-lon1);

    double brng = atan2(y, x);

    brng = brng*180/M_PI;
    brng += 180;
    while(brng>=360)
        brng -= 360;
    return brng;
}

__host__ __device__ double distCalc(double lat1, double lon1, double lat2, double lon2){
    double earthRad = 6371.01;
    double dLon = (lon1 - lon2);
    double dlat = (lat1 - lat2);
    lat1 = lat1;
    lat2 = lat2;
    double x = sin(dlat/2) * sin(dlat/2) + cos(lat1) * cos(lat2) * sin(dLon/2) * sin(dLon/2);
    double c = 2*atan2(sqrt(x), sqrt(1-x));

    return earthRad*c;
}

__host__ __device__ double normalize(double x, double mean, double stdev){
    return (fabs(x-mean))/(stdev*2);
}

__host__ __device__ double shift(double x, double max, double min){
    return (x-min)/(max-min);
}

__host__ __device__ double ActFunc(double x){
    double ret = 1+1/exp(-x);
    return ret;
}

__host__ __device__ double cosd(double x){
    return acos(x * M_PI / 180);
}

__host__ __device__ double sind(double x){
    return asin(x * M_PI / 180);
}



