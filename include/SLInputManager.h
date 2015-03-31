//#############################################################################
//  File:      SLInputManager.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLINPUTMANAGER_H
#define SLINPUTMANAGER_H

#include <stdafx.h>
#include <SLInputEvent.h>
#include <SLInputDevice.h>

//-----------------------------------------------------------------------------
//! SLInputManager. manages system input and custom input devices.
/*! Every user input has to go through the SLInputManager. System event's
like touch, mouse, character input will be encapsulated in SLInputEvent
subclasses and will be queued up before being sent to the relevant SLSceneView.
Custom SLInputDevices can also be created. The SLInputDevices are guaranteed to 
receive a call to their poll() function whenever the SLInputManager requires them
to send out new events.

SLInputManager is a singleton class and only ever exists once.
*/
class SLInputManager
{
friend class SLInputDevice;

public:
    static  SLInputManager& instance        ();

            SLbool          pollEvents      ();
            void            queueEvent      (const SLInputEvent* e);

private:
    static  SLInputManager  _instance;      //!< the singleton instance of the input manager
            SLQInputEvent   _systemEvents;  //!< queue for known system events
            SLVInputDevice  _devices;       //!< list of activated SLInputDevices

                            // Constructor is private to prevent instantiation
                            SLInputManager  (){ }

            SLbool          processQueuedEvents();
};
//-----------------------------------------------------------------------------
#endif
