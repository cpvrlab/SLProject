#include "SENSGps.h"
#include <Utils.h>
#include <algorithm>

SENSGps::Location SENSGps::getLocation() const
{
    const std::lock_guard<std::mutex> lock(_llaMutex);
    return _location;
}

void SENSGps::setLocation(SENSGps::Location location)
{
    //estimate time before running into lock
    SENSTimePt timePt = SENSClock::now();

    {
        const std::lock_guard<std::mutex> lock(_llaMutex);
        _location = location;
        _timePt   = timePt;
    }

    {
        std::lock_guard<std::mutex> lock(_listenerMutex);
        for (SENSGpsListener* l : _listeners)
            l->onGps(timePt, location);
    }
}

void SENSGps::registerListener(SENSGpsListener* listener)
{
    std::lock_guard<std::mutex> lock(_listenerMutex);
    if (std::find(_listeners.begin(), _listeners.end(), listener) == _listeners.end())
        _listeners.push_back(listener);
}

void SENSGps::unregisterListener(SENSGpsListener* listener)
{
    std::lock_guard<std::mutex> lock(_listenerMutex);
    for (auto it = _listeners.begin(); it != _listeners.end(); ++it)
    {
        if (*it == listener)
        {
            _listeners.erase(it);
            break;
        }
    }
}

bool SENSGps::permissionGranted() const
{
    return _permissionGranted;
}

void SENSGps::registerPermissionListener(std::function<void(void)> listener)
{
    {
        std::lock_guard<std::mutex> lock(_permissionListenerMutex);
        _permissionListeners.push_back(listener);
    }
    //inform about initial state
    informPermissionListeners();
}

void SENSGps::informPermissionListeners()
{
    std::lock_guard<std::mutex> lock(_permissionListenerMutex);
    for (auto listener : _permissionListeners)
        listener();
}

void SENSGps::updatePermission(bool granted)
{
    _permissionGranted = granted;
    informPermissionListeners();
}

//-----------------------------------------------------------------------------

SENSDummyGps::~SENSDummyGps()
{
    stopSimulation();
}

bool SENSDummyGps::start()
{
    if (_running)
        return false;

    Utils::log("SENSDummyGps", "start");
    startSimulation();
    _running = true;
    return true;
}

void SENSDummyGps::stop()
{
    Utils::log("SENSDummyGps", "stop");
    stopSimulation();
    _running = false;
}

void SENSDummyGps::startSimulation()
{
    _stop   = false;
    _thread = std::thread(&SENSDummyGps::run, this);
}

void SENSDummyGps::stopSimulation()
{
    _stop = true;
    if (_thread.joinable())
        _thread.join();
}

void SENSDummyGps::run()
{
    int i = 0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (_stop)
            break;

        i++;
        i = i % _dummyLocs.size();

        setLocation(_dummyLocs[i]);
    }
}
