#include <kernelDefs.h>

__constant__ int inputData[15*3*100]{0};
__constant__ double answers[7];
__constant__ double globalQuakes[5];
__constant__ double siteData[15*2]{0};
__constant__ double Kp;
__constant__ int site_offset[15]{0};
__constant__ int channel_offset[3];
__constant__ int trainingsize;
__constant__ devicePair<dcon, dcon> connections[200];
