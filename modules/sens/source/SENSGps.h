#ifndef SENS_GPS_H
#define SENS_GPS_H

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <functional>

#include "SENS.h"

class SENSGpsListener;

class SENSGps
{
public:
    struct Location
    {
        double latitudeDEG  = 0;
        double longitudeDEG = 0;
        double altitudeM    = 0;
        float  accuracyM    = -1.f;
    };

    virtual ~SENSGps() {}
    //start gps sensor
    virtual bool start() = 0;
    virtual void stop()  = 0;

    Location getLocation() const;

    bool isRunning() { return _running; }

    bool permissionGranted() const;

    void registerListener(SENSGpsListener* listener);
    void unregisterListener(SENSGpsListener* listener);

    void registerPermissionListener(std::function<void(void)> listener);
    void updatePermission(bool granted);

protected:
    void setLocation(SENSGps::Location location);
    void informPermissionListeners();

    bool             _running = false;
    std::atomic_bool _permissionGranted{false};

private:
    SENSTimePt         _timePt;
    Location           _location;
    mutable std::mutex _llaMutex;

    std::vector<SENSGpsListener*> _listeners;
    std::mutex                    _listenerMutex;

    mutable std::mutex                     _permissionListenerMutex;
    std::vector<std::function<void(void)>> _permissionListeners;
};

class SENSGpsListener
{
public:
    virtual ~SENSGpsListener() {}
    virtual void onGps(const SENSTimePt& timePt, const SENSGps::Location& loc) = 0;
};

class SENSDummyGps : public SENSGps
{
public:
    SENSDummyGps()
    {
        _permissionGranted = true;
    }
    ~SENSDummyGps();

    bool start() override;
    void stop() override;

    void addDummyPos(SENSGps::Location loc)
    {
        _dummyLocs.push_back(loc);
    }

private:
    void startSimulation();
    void stopSimulation();
    void run();

    std::thread      _thread;
    std::atomic_bool _stop{false};

    std::vector<SENSGps::Location> _dummyLocs;
};

#endif
