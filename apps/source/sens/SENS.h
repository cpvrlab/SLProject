#ifndef SENS_H
#define SENS_H

#include <chrono>
#include <Utils.h>

enum class SENSType
{
    CAM,
    VIDEO,
    GPS,
    ORIENTATION
};

using SENSClock        = std::chrono::high_resolution_clock;
using SENSTimePt       = std::chrono::high_resolution_clock::time_point;
using SENSMicroseconds = std::chrono::microseconds;

//#define SENS_DEBUG(...) Utils::log("SENS DEBUG", __VA_ARGS__);
#define SENS_DEBUG
#define SENS_INFO(...) Utils::log("SENS INFO", __VA_ARGS__);
#define SENS_WARN(...) Utils::log("SENS WARN", __VA_ARGS__);

#endif //SENS_H
