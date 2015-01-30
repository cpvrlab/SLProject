//#############################################################################
//  File:      SLInputEvent.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SLINPUTEVENT_H
#define SLINPUTEVENT_H

#include <stdafx.h>
#include <SLEnums.h>

class SLInputEvent
{
public:
    enum Type
    {
        SLCommand,
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
        NumEvents
    } type;         //!< concrete type of the event
    SLint svIndex;  //!< index of the receiving scene view for this event

	SLInputEvent(Type t)
        : type(t)
	{ }
};


class SLMouseEvent : public SLInputEvent
{
public:
    SLint x;
    SLint y;
    SLMouseButton button;
    SLKey modifier;

    SLMouseEvent(Type t)
        : SLInputEvent(t)
    { }
};

class SLKeyEvent : public SLInputEvent
{
public:
    SLKey key;
    SLKey modifier;

    SLKeyEvent(Type t)
        : SLInputEvent(t)
    { }
};

class SLTouchEvent : public SLInputEvent
{
public:
    SLint x1;
    SLint y1;
    SLint x2;
    SLint y2;

    SLTouchEvent(Type t)
        : SLInputEvent(t)
    { }
};

class SLRotationEvent : public SLInputEvent
{
public:
    float x, y, z, w;

    SLRotationEvent(Type t)
        : SLInputEvent(t)
    { }
};

class SLResizeEvent : public SLInputEvent
{
public:
    int width;
    int height;
    
    SLResizeEvent()
        : SLInputEvent(Resize)
    { }
};

class SLCommandEvent : public SLInputEvent
{
public:
    SLCmd cmd;

    SLCommandEvent()
        : SLInputEvent(SLCommand)
    { }
};

typedef std::queue<const SLInputEvent*> SLInputEventPtrQueue;

//-----------------------------------------------------------------------------
#endif