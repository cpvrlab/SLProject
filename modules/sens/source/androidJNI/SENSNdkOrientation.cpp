#include "SENSNdkOrientation.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>

bool SENSNdkOrientation::start()
{
    _running = true;
    return true;
}

void SENSNdkOrientation::stop()
{
    if (!_running)
        return;
    _running = false;
}

void SENSNdkOrientation::updateOrientation(const SENSOrientation::Quat& orientation)
{
    if (_running)
        setOrientation(orientation);
}
