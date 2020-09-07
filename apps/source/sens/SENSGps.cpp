#include "SENSGps.h"
#include <Utils.h>

SENSGps::Location SENSGps::getLocation()
{
    const std::lock_guard<std::mutex> lock(_llaMutex);
    return _location;
}

void SENSGps::setLocation(SENSGps::Location location)
{
    const std::lock_guard<std::mutex> lock(_llaMutex);
    _location = location;
}

/**********************************************************************/

SENSDummyGps::~SENSDummyGps()
{
    stopSimulation();
}

bool SENSDummyGps::start()
{
    Utils::log("SENSDummyGps", "start");
    startSimulation();
    _running = true;
    return _running;
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

        Utils::log("SENSDummyGps", "run");
        setLocation(_dummyLocs[i]);
    }
}
