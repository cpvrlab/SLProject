#include "SENSOrientation.h"

SENSOrientation::Quat SENSOrientation::getOrientation()
{
    const std::lock_guard<std::mutex> lock(_orientationMutex);
    return _orientation;
}

void SENSOrientation::setOrientation(SENSOrientation::Quat orientation)
{
    const std::lock_guard<std::mutex> lock(_orientationMutex);
    _orientation = orientation;
}
