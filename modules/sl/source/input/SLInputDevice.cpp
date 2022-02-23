//#############################################################################
//  File:      SLInputDevice.cpp
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <algorithm>
#include <SLInputManager.h>

//-----------------------------------------------------------------------------
/*! Constructor for SLInputDevices. This will automatically enable the device,
adding them to the SLInputManager.
 */
SLInputDevice::SLInputDevice(SLInputManager& inputManager)
  : _inputManager(inputManager)
{
    // enable any input device on creation
    enable();
}
//-----------------------------------------------------------------------------
/*! The destructor removes the device from SLInputManager again if necessary.
 */
SLInputDevice::~SLInputDevice()
{
    disable();
}
//-----------------------------------------------------------------------------
/*! Enabling an SLInputDevice will add it to the device list kept by
SLInputManager
*/
void SLInputDevice::enable()
{
    _inputManager.devices().push_back(this);
}
//-----------------------------------------------------------------------------
/*! Enabling an SLInputDevice will remove it from the device list kept by
SLInputManager
*/
void SLInputDevice::disable()
{
    SLVInputDevice& dl = _inputManager.devices();
    dl.erase(std::remove(dl.begin(), dl.end(), this), dl.end());
}
//-----------------------------------------------------------------------------
