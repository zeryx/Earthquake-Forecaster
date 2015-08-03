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
    std::istringstream(val) >> start.tm_yday;
    std::getline(sstart, val, ':');
    std::istringstream(val) >> start.tm_hour;
    std::getline(sstart, val, ':');
    std::getline(sstart, val, ':');
    std::istringstream(val) >> start.tm_min;
    std::getline(sstart, val, ':');
    std::getline(sstart, val);
    std::istringstream(val) >> start.tm_sec;

    std::getline(sstop, val, '-');
    std::istringstream(val) >> stop.tm_year;
    std::getline(sstop, val, '-');
    std::istringstream(val) >> stop.tm_mon;
    std::getline(sstop, val, ' ');
    std::istringstream(val) >> stop.tm_yday;
    std::getline(sstop, val, ':');
    std::istringstream(val) >> stop.tm_hour;
    std::getline(sstop, val, ':');
    std::getline(sstop, val, ':');
    std::istringstream(val) >> stop.tm_min;
    std::getline(sstop, val, ':');
    std::getline(sstop, val);
    std::istringstream(val) >> stop.tm_sec;
    time_t start_time = std::mktime(&start);
    time_t stop_time = std::mktime(&stop);
    long diff_time = stop_time - start_time;
    return diff_time/3600;

}
#endif
