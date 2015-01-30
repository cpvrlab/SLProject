//#############################################################################
//  File:      SLInputDevice.cpp
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLInputManager.h>
#include <SLInputDevice.h>


SLInputDevice::SLInputDevice()
{
    // enable any input device on creation
    // @todo is this good practice?
    enable();
}

SLInputDevice::~SLInputDevice()
{
    disable();
}

void SLInputDevice::enable()
{
    SLInputManager::instance()._devices.push_back(this);
}
void SLInputDevice::disable()
{
    SLVInputDevice& dl = SLInputManager::instance()._devices;
    dl.erase(remove(dl.begin(), dl.end(), this), dl.end());
}