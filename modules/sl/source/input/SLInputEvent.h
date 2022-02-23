//#############################################################################
//  File:      SLInputEvent.h
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLINPUTEVENT_H
#define SLINPUTEVENT_H

#include <SL.h>
#include <SLEnums.h>
#include <queue>

//-----------------------------------------------------------------------------
//! Baseclass for all system input events.
/*! SLProject has it's own internal event queue to guarantee the same input
handling accross multiple platforms. Some system's might send system events
asynchronously. This is why we provide the SLInputEvent class for all
system relevant input events.
*/
class SLInputEvent
{
public:
    enum Type
    {
        MouseMove,
        MouseDown,
        MouseUp,
        MouseDoubleClick,
        MouseWheel,
        Touch2Move,
        Touch2Down,
        Touch2Up,
        KeyDown,
        KeyUp,
        Resize,
        DeviceRotationPYR,
        DeviceRotationQUAT,
        CharInput,
        ScrCapture,
        NumEvents
    } type;        //!< concrete type of the event
    SLint svIndex; //!< index of the receiving scene view for this event

    SLInputEvent(Type t) : type(t) {}
};

//-----------------------------------------------------------------------------
//! Specialized SLInput class for all mouse related input events.
class SLMouseEvent : public SLInputEvent
{
public:
    SLint         x;
    SLint         y;
    SLMouseButton button;
    SLKey         modifier;

    SLMouseEvent(Type t) : SLInputEvent(t) {}
};

//-----------------------------------------------------------------------------
//! Specialized SLInput class for all keypress related input events.
class SLKeyEvent : public SLInputEvent
{
public:
    SLKey key;
    SLKey modifier;

    SLKeyEvent(Type t) : SLInputEvent(t) {}
};

//-----------------------------------------------------------------------------
//! Specialized SLInput class for touch related input events.
class SLTouchEvent : public SLInputEvent
{
public:
    SLint x1;
    SLint y1;
    SLint x2;
    SLint y2;

    SLTouchEvent(Type t) : SLInputEvent(t) {}
};

//-----------------------------------------------------------------------------
//! Specialized SLInput class for all device rotation related input events.
class SLRotationEvent : public SLInputEvent
{
public:
    float x, y, z, w;

    SLRotationEvent(Type t) : SLInputEvent(t) {}
};

//-----------------------------------------------------------------------------
//! Specialized SLInput class for window resize events.
class SLResizeEvent : public SLInputEvent
{
public:
    int width;
    int height;

    SLResizeEvent() : SLInputEvent(Resize) {}
};

//-----------------------------------------------------------------------------
//! Specialized SLInput class for unicode character input.
/*! Character input differs from simple key input that it can be generated from
a combination of different key presses. Some key's might not fire a character
event, others might fire multiple at once.
*/
class SLCharInputEvent : public SLInputEvent
{
public:
    SLuint character;

    SLCharInputEvent() : SLInputEvent(CharInput) {}
};

//-----------------------------------------------------------------------------
//! Specialized SLInput class to trigger a screen capture
class SLScrCaptureRequestEvent : public SLInputEvent
{
public:
    SLstring path;

    SLScrCaptureRequestEvent() : SLInputEvent(ScrCapture) {}
};

//-----------------------------------------------------------------------------
typedef std::queue<const SLInputEvent*> SLQInputEvent;
//-----------------------------------------------------------------------------
#endif
