//#############################################################################
//  File:      SLInputManager.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLINPUTMANAGER_H
#define SLINPUTMANAGER_H

#include <SLInputDevice.h>
#include <SLInputEvent.h>

class SLScene;
//-----------------------------------------------------------------------------
//! SLInputManager. manages system input and custom input devices.
/*!  One static instance of SLInputManager is used in SLApplication. Every user
 input has to go through the SLInputManager. System event's like touch, mouse,
 character input will be encapsulated in SLInputEvent subclasses and will be
 queued up before being sent to the relevant SLSceneView.
 Custom SLInputDevices can also be created. The SLInputDevices are guaranteed to
 receive a call to their poll() function whenever the SLInputManager requires
 them to send out new events. The method pollAndProcessEvents is called every
 frame in SLScene::onUpdate.
*/
class SLInputManager
{
    friend class SLInputDevice;

public:
    SLInputManager() { ; }

    SLbool          pollAndProcessEvents(SLScene* s);
    void            queueEvent(const SLInputEvent* e);
    SLVInputDevice& devices() { return _devices; }

private:
    SLQInputEvent  _systemEvents; //!< queue for known system events
    SLVInputDevice _devices;      //!< list of activated SLInputDevices

    SLbool processQueuedEvents(SLScene* s);
};
//-----------------------------------------------------------------------------
#endif
