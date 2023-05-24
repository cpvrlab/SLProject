#ifndef SENS_ORIENTATION_H
#define SENS_ORIENTATION_H

#include <mutex>
#include <atomic>
#include <vector>
#include <thread>
#include <Utils.h>

#include "SENS.h"

class SENSOrientationListener;

/*

 Android sensor coordinate system:
 (https://developer.android.com/guide/topics/sensors/sensors_overview)

 Up = z   North = y
      |  /
      | /
      |/
      +------ East = x
      +---------+
     / +-----+ /
    / /     / /
   / /     / /
  / +-----+ /
 /    0    /
+---------+

 iOS sensor coordinate system:
 (https://developer.apple.com/documentation/coremotion/getting_processed_device-motion_data/understanding_reference_frames_and_device_attitude)
 In iOS we configure CMMotionManager with xMagneticNorthZVertical which means its a frame, where x points north, y points west and z points up (NWU).
 In the iOS code, we add rotation of 90 deg. around z-axis to relate the sensor rotation to an ENU-frame (as in Android).

 Up = z   West = y
      |  /
      | /
      |/
      +------ North = x
      +---------+
     / +-----+ /
    / /     / /
   / /     / /
  / +-----+ /
 /    0    /
+---------+

 */

class SENSOrientation
{
public:
    struct Quat
    {
        Quat() = default;
        Quat(float x, float y, float z, float w)
          : quatX(x),
            quatY(y),
            quatZ(z),
            quatW(w)
        {
        }
        float quatX = 0.f;
        float quatY = 0.f;
        float quatZ = 0.f;
        float quatW = 0.f;
    };

    virtual ~SENSOrientation() {}
    //start gps sensor
    virtual bool start() = 0;
    virtual void stop()  = 0;

    Quat getOrientation() const;

    bool isRunning() { return _running; }

    void registerListener(SENSOrientationListener* listener);
    void unregisterListener(SENSOrientationListener* listener);

protected:
    void setOrientation(Quat orientation);

    bool _running = false;

private:
    SENSTimePt         _timePt;
    Quat               _orientation;
    mutable std::mutex _orientationMutex;

    std::vector<SENSOrientationListener*> _listeners;
    std::mutex                            _listenerMutex;
};

class SENSOrientationListener
{
public:
    virtual ~SENSOrientationListener() {}
    virtual void onOrientation(const SENSTimePt& timePt, const SENSOrientation::Quat& ori) = 0;
};

class SENSDummyOrientation : public SENSOrientation
{
public:
    ~SENSDummyOrientation() override {}
    void setDummyQuat(SENSOrientation::Quat quat)
    {
        setOrientation(quat);
    }

    bool start() override
    {
        _running = true;
        return _running;
    }

    void stop() override
    {
    }
};

#endif
