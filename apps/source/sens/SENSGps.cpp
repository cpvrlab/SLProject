#include "SENSGps.h"
#include <Utils.h>

SENSGps::Location SENSGps::getLocation()
{
    const std::lock_guard<std::mutex> lock(_llaMutex);
    return _location;
}

void SENSGps::setLocation(double latitudeDEG,
                          double longitudeDEG,
                          double altitudeM,
                          float  accuracyM)
{
    const std::lock_guard<std::mutex> lock(_llaMutex);
    _location.latitudeDEG  = latitudeDEG;
    _location.longitudeDEG = longitudeDEG;
    _location.altitudeM    = altitudeM;
    _location.accuracyM    = accuracyM;
}

/**********************************************************************/

SENSDummyGps::SENSDummyGps(double latitudeDEG,
                           double longitudeDEG,
                           double altitudeM)
{
    _dummyLoc.latitudeDEG  = latitudeDEG;
    _dummyLoc.longitudeDEG = longitudeDEG;
    _dummyLoc.altitudeM    = altitudeM;
}

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
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (_stop)
            break;

        Utils::log("SENSDummyGps", "run");
        setLocation(_dummyLoc.latitudeDEG,
                    _dummyLoc.longitudeDEG,
                    _dummyLoc.altitudeM,
                    _dummyLoc.accuracyM);
    }
}
