#include <kernelDefs.h>

__host__ __device__ double bearingCalc(double lat1, double lon1, double lat2, double lon2){
    double dLon = (lon2 - lon1);

    double y = asin(dLon) * acos(lat2);
    double x = acos(lat1) * asin(lat2) - asin(lat1) * acos(lat2) * acos(dLon);

    double brng = atan2(y, x);

    brng = brng*M_PI/180;
    brng += 360;
    while(brng>= 360)
        brng -= 360;
    brng = 360 - brng;

    return brng;
}

__host__ __device__ double distCalc(double lat1, double lon1, double lat2, double lon2){
    double earthRad = 6371.01;
    double deltalon = abs(lon1 - lon2);
    if(deltalon > 180)
        deltalon = 360 - deltalon;
    double ret = earthRad * atan2( sqrt( pow( cosd(lat1) * sind(deltalon), 2) +
                                         pow( cosd(lat2) * sind(lat1) - sind(lat2) * cosd(lat1) * cosd(deltalon), 2) ),
                                   sind(lat2) * sind(lat1) + cosd(lat2) * cosd(lat1) * cosd(deltalon));
    return ret;
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



