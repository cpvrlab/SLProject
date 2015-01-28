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
    SLInputManager::instance()._devices.push_back(this);
}
