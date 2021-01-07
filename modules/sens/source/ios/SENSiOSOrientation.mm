#include "SENSiOSOrientation.h"

SENSiOSOrientation::SENSiOSOrientation()
{
    _orientationDelegate = [[SENSiOSOrientationDelegate alloc] init];
    //set update callback
    [_orientationDelegate setUpdateCB:std::bind(&SENSiOSOrientation::updateOrientation, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)];
}

bool SENSiOSOrientation::start()
{
    _running = [_orientationDelegate start];
    return _running;
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
