#include "SENSiOSGps.h"

SENSiOSGps::SENSiOSGps()
{
    _gpsDelegate = [[SENSiOSGpsDelegate alloc] init];
    //set update callback
    [_gpsDelegate setUpdateCB:std::bind(&SENSiOSGps::updateLocation, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)];
    //set permission callback
    [_gpsDelegate setPermissionCB:std::bind(&SENSGps::updatePermission, this, std::placeholders::_1)];
}

bool SENSiOSGps::start()
{
    if (!_permissionGranted)
        return false;
    else
    {
        _running = [_gpsDelegate start];
        return _running;
    }
}

void SENSiOSGps::stop()
{
    [_gpsDelegate stop];
    _running = false;
}

void SENSiOSGps::updateLocation(double latitudeDEG,
                                double longitudeDEG,
                                double altitudeM,
                                double accuracyM)
{
    setLocation({latitudeDEG, longitudeDEG, altitudeM, (float)accuracyM});
}

void SENSiOSGps::askPermission()
{
    [_gpsDelegate askPermission];
}
