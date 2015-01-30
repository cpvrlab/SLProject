//#############################################################################
//  File:      SLInputManager.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLINPUTMANAGER_H
#define SLINPUTMANAGER_H

#include <stdafx.h>
#include <SLInputEvent.h>
#include <SLInputDevice.h>

//-----------------------------------------------------------------------------
//! Simple input manager, for now a singleton that handles the polling of SLInputDevices
class SLInputManager
{
friend class SLInputDevice;

public:
    ~SLInputManager();
    static SLInputManager& instance();

    void update();

    void queueEvent(const SLInputEvent* e);

private:
    static SLInputManager _instance;

    SLInputEventPtrQueue  _systemEventQueue;
    SLVInputDevice _devices;

    // prevent instantiation
    SLInputManager()
    { }

    void processQueuedEvents();
};


#endif