#ifndef SENS_ORIENTATION_H
#define SENS_ORIENTATION_H

#include <mutex>
#include <atomic>
#include <vector>
#include <thread>
#include <Utils.h>

#include "SENS.h"

class SENSOrientationListener;

class SENSOrientation
{
public:
    struct Quat
    {
        float quatX = 0.f;
        float quatY = 0.f;
        float quatZ = 0.f;
        float quatW = 0.f;
    };

    virtual ~SENSOrientation() {}
    //start gps sensor
    virtual bool start() = 0;
    virtual void stop()  = 0;

    Quat getOrientation();

    bool isRunning() { return _running; }

    void registerListener(SENSOrientationListener* listener);
    void unregisterListener(SENSOrientationListener* listener);

protected:
    void setOrientation(Quat orientation);

    bool _running = false;

private:
    SENSTimePt _timePt;
    Quat       _orientation;
    std::mutex _orientationMutex;

    std::vector<SENSOrientationListener*> _listeners;
    std::mutex                            _listenerMutex;
};

class SENSOrientationListener
{
public:
    virtual ~SENSOrientationListener() {}
    virtual void onOrientation(const SENSTimePt& timePt, const SENSOrientation::Quat& ori) = 0;
};

#endif
