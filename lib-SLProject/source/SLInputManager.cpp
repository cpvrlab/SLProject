//#############################################################################
//  File:      SLInputManager.cpp
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLInputManager.h>

//-----------------------------------------------------------------------------
//! static instance variable declaration
SLInputManager SLInputManager::_instance;

//-----------------------------------------------------------------------------
/*! static accessor to the singleton instance of this class */
SLInputManager& SLInputManager::instance()
{
    return _instance;
}

//-----------------------------------------------------------------------------
/*! Sends any queued up system event's to their correct receiver and
polls all activated SLInputDevices. 

@note   The event queue is similar to how Qt manages it's events. The main difference
        is, that we don't use the SLInputEvent class outside of the SLInputManager.
        The SLInputManager calls the correct SLSceneView input handler functions directly.
        Also we don't allow for custom SLInputEvents. This is the other main difference
        to the Qt event system. The decision to go this route is simplicity for now. 
        It is totally sufficient for our use cases to provide the user with the
        SLInputDevice interface to realize custom input. 
        However it has to be considered, that Qt also has many GUI related events like
        MouseEnter, MouseLeave, Drag etc. For a sophisticated GUI implementation the 
        whole input management in SL would have to be reviewed.
*/
SLbool SLInputManager::pollEvents()
{
    // process system events first
    SLbool consumedEvents = processQueuedEvents();

    // process custom input devices
    for(auto device : _devices)
        consumedEvents |= device->poll();

    return consumedEvents;
}

//-----------------------------------------------------------------------------
/*! Add a new SLInputEvent to the event queue. The queue will be emtied when
a call to SLInputManager::pollEvents is made. The passed in SLInputEvents have 
to be dynamically allocated by the user, the deallocation is handled by the
SLInputManager */
void SLInputManager::queueEvent(const SLInputEvent* e)
{
    _systemEvents.push(e);
}

//-----------------------------------------------------------------------------
/*! Work off any queued up input event's and notify the correct receiver.
@note   this is similar to the Qt QObject::event function.*/
SLbool SLInputManager::processQueuedEvents()
{
    SLQInputEvent& q = _systemEvents;

    // flag if an event has been consumed by a receiver
    SLbool eventConsumed = false;

    while (!q.empty())
    {
        const SLInputEvent* e = q.front();
        q.pop();

        SLSceneView* sv = SLScene::current->sv(e->svIndex);
        
        if (sv)
        {   switch (e->type)
            {
                case SLInputEvent::Command:            { const SLCommandEvent* ce = (const SLCommandEvent*)e; eventConsumed |= sv->onCommand(ce->cmd); } break;

                case SLInputEvent::MouseMove:          { const SLMouseEvent* me = (const SLMouseEvent*)e; eventConsumed |= sv->onMouseMove(me->x, me->y); } break;
                case SLInputEvent::MouseDown:          { const SLMouseEvent* me = (const SLMouseEvent*)e; eventConsumed |= sv->onMouseDown(me->button, me->x, me->y, me->modifier); } break;
                case SLInputEvent::MouseUp:            { const SLMouseEvent* me = (const SLMouseEvent*)e; eventConsumed |= sv->onMouseUp(me->button, me->x, me->y, me->modifier); } break;
                case SLInputEvent::MouseDoubleClick:   { const SLMouseEvent* me = (const SLMouseEvent*)e; eventConsumed |= sv->onDoubleClick(me->button, me->x, me->y, me->modifier); } break;
                case SLInputEvent::MouseWheel:         { const SLMouseEvent* me = (const SLMouseEvent*)e; eventConsumed |= sv->onMouseWheel(me->y, me->modifier); } break;
                case SLInputEvent::LongTouch:          { const SLMouseEvent* me = (const SLMouseEvent*)e; eventConsumed |= sv->onLongTouch(me->x, me->y); } break;

                case SLInputEvent::Touch2Move:         { const SLTouchEvent* te = (const SLTouchEvent*)e; eventConsumed |= sv->onTouch2Move(te->x1, te->y1, te->x2, te->y2); } break;
                case SLInputEvent::Touch2Down:         { const SLTouchEvent* te = (const SLTouchEvent*)e; eventConsumed |= sv->onTouch2Down(te->x1, te->y1, te->x2, te->y2); } break;
                case SLInputEvent::Touch2Up:           { const SLTouchEvent* te = (const SLTouchEvent*)e; eventConsumed |= sv->onTouch2Up(te->x1, te->y1, te->x2, te->y2); } break;

                case SLInputEvent::KeyDown:            { const SLKeyEvent* ke = (const SLKeyEvent*)e; eventConsumed |= sv->onKeyPress(ke->key, ke->modifier); } break;
                case SLInputEvent::KeyUp:              { const SLKeyEvent* ke = (const SLKeyEvent*)e; eventConsumed |= sv->onKeyRelease(ke->key, ke->modifier); } break;

                case SLInputEvent::Resize:             { const SLResizeEvent* re = (const SLResizeEvent*)e; sv->onResize(re->width, re->height); } break;

                case SLInputEvent::DeviceRotationPYR:  { const SLRotationEvent* re = (const SLRotationEvent*)e; sv->onRotationPYR(re->x, re->y, re->z, 3.0f); } break;
                case SLInputEvent::DeviceRotationQUAT: { const SLRotationEvent* re = (const SLRotationEvent*)e; sv->onRotationQUAT(re->x, re->y, re->z, re->w); } break;
            }
        }

        delete e;
    }

    return eventConsumed;
}
