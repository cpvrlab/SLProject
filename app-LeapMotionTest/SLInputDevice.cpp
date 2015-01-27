
#include <stdafx.h>
#include <SLInputManager.h>
#include <SLInputDevice.h>


SLInputDevice::SLInputDevice()
{
    SLInputManager::instance()._devices.push_back(this);
}
