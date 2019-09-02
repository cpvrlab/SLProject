#ifndef WAI_H
#define WAI_H

#include <map>

#include <WAIHelper.h>
#include <WAIMath.h>
#include <WAISensor.h>
#include <WAIMode.h>
#include <WAIModeAruco.h>
#include <WAIModeOrbSlam2.h>
#include <WAIModeOrbSlam2DataOriented.h>

namespace WAI
{

class WAI_API WAI
{
    public:
    WAI(std::string dataRoot);
    void  setDataRoot(std::string dataRoot);
    void  activateSensor(SensorType sensorType, void* sensorInfo);
    void  updateSensor(SensorType type, void* value);
    bool  whereAmI(cv::Mat* pose);
    Mode* setMode(ModeType mode);
    Mode* getCurrentMode();
    ModeType getCurrentModeType();

    private:
    std::string                   _dataRoot;
    Mode*                         _mode = nullptr;
    ModeType                      _modeType;
    std::map<SensorType, Sensor*> _sensors;
};
}

#endif
