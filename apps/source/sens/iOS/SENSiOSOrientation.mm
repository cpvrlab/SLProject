#include "SENSiOSOrientation.h"

SENSiOSOrientation::SENSiOSOrientation()
{
    _orientationDelegate = [[SENSiOSOrientationDelegate alloc] init];
    //set update callback
    [_orientationDelegate setUpdateCB:std::bind(&SENSiOSOrientation::updateOrientation, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)];
    //set permission callback
    [_orientationDelegate setPermissionCB:std::bind(&SENSiOSOrientation::updatePermission, this, std::placeholders::_1)];
    
    _permissionGranted = true;
}

bool SENSiOSOrientation::start()
{
    if (!_permissionGranted)
        return false;
    else
    {
        _running = [_orientationDelegate start];
        return _running;
    }
}

void SENSiOSOrientation::stop()
{
    [_orientationDelegate stop];
    _running = false;
}

void SENSiOSOrientation::updateOrientation(float quatX,
                                           float quatY,
                                           float quatZ,
                                           float quatW)
{
    setOrientation({quatX, quatY, quatZ, quatW});
}

void SENSiOSOrientation::updatePermission(bool granted)
{
    _permissionGranted = granted;
}
