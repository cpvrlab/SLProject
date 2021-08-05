#include "SENSAndroidOrientation.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>

bool SENSAndroidOrientation::start()
{
    _running = true;
    return true;
}

void SENSAndroidOrientation::stop()
{
    if (!_running)
        return;
    _running = false;
}

void SENSAndroidOrientation::updateOrientation(const SENSOrientation::Quat& orientation)
{
    if (_running)
        setOrientation(orientation);
}
