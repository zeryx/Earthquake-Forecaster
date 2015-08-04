#ifndef DATEDIFF_H
#define DATEDIFF_H
#include <string>
#include <sstream>
#include <iostream>
#include <ctime>
int timeDifferenceCalculation(std::string startStr, std::string stopStr){
    std::tm start, stop;
    std::stringstream sstart(startStr), sstop(stopStr);
    std::string val;
    std::getline(sstart, val, '-');
    std::istringstream(val) >> start.tm_year;
    std::getline(sstart, val, '-');
    std::istringstream(val) >> start.tm_mon;
    std::getline(sstart, val, ' ');
    std::istringstream(val) >> start.tm_mday;
    std::getline(sstart, val, ':');
    std::istringstream(val) >> start.tm_hour;
    std::getline(sstart, val, ':');
    std::istringstream(val) >> start.tm_min;

    std::getline(sstop, val, '-');
    std::istringstream(val) >> stop.tm_year;
    std::getline(sstop, val, '-');
    std::istringstream(val) >> stop.tm_mon;
    std::getline(sstop, val, ' ');
    std::istringstream(val) >> stop.tm_mday;
    std::getline(sstop, val, ':');
    std::istringstream(val) >> stop.tm_hour;
    std::getline(sstop, val, ':');
    std::istringstream(val) >> stop.tm_min;

    time_t start_time = std::mktime(&start);
    time_t stop_time = std::mktime(&stop);
    int diff_time = stop_time - start_time;
    diff_time = diff_time/36000;
    std::cout<<start.tm_year<<std::endl;
    std::cout<<start.tm_mon<<std::endl;
    std::cout<<start.tm_mday<<std::endl;
    std::cout<<start.tm_hour<<std::endl;
    std::cout<<start.tm_min<<std::endl;
    std::cout<<diff_time<<std::endl;
    return diff_time;

}
#endif
