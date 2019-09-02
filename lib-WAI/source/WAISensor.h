#ifndef WAI_SENSOR_H
#define WAI_SENSOR_H

#include <WAIMode.h>

namespace WAI
{
enum SensorType
{
    SensorType_None,
    SensorType_Camera
};

class Sensor
{
    public:
    virtual void update(void* data) = 0;
};
}

#endif
