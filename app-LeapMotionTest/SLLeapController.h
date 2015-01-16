//  File:      SLLeapController.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2015 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLEAPCONTROLLER_H
#define SLLEAPCONTROLLER_H

#include <stdafx.h>
#include <Leap.h>
#include <SLLeapFinger.h>
#include <SLLeapHand.h>
#include <SLLeapTool.h>
#include <SLLeapGesture.h>


// @note    we could slap togeter a simple c++ delegate if we could use c++11
//          which would allow us to not have to define these listener interfaces.

// listener interfaces
class SLLeapGestureListener
{
friend class SLLeapController;
protected:
    virtual void onLeapGesture(const SLLeapGesture& gesture) = 0;
};

class SLLeapHandListener
{
friend class SLLeapController;
protected:
    virtual void onLeapHandChange(const vector<SLLeapHand>& hands) = 0;
};

class SLLeapToolListener
{
friend class SLLeapController;
protected:
    virtual void onLeapToolChange(const vector<SLLeapTool>& tools) = 0;
};

// @note    below is a good example of a smart ptr usecase
typedef vector<SLLeapGestureListener*> SLVLeapGestureListenerPtr;
typedef vector<SLLeapHandListener*> SLVLeapHandListenerPtr;
typedef vector<SLLeapToolListener*> SLVLeapToolListenerPtr;

// We use protected inheritance to not expose the leap listener interface
// but to allow for future public inheritance of the SLLeapController
/// @todo   Test if the Leap::Listener onFrame function gets called more often than an image is rendered.
///         If this is the case we need to mitigate that by utilizing an event queue. We queue all gesture
///         events up and when a new frame update happens we contact the gesture listeners with all events.
///         For  the hand listeners we only ever need the newest hand status, same for tools, but we should
///         look into it still.
class SLLeapController : public Leap::Listener
{
public:
    SLLeapController();
    ~SLLeapController();
    
    void registerGestureListener(SLLeapGestureListener* listener);
    void registerHandListener(SLLeapHandListener* listener);
    void registerToolListener(SLLeapToolListener* listener);
    
    void removeGestureListener(SLLeapGestureListener* listener);
    void removeHandListener(SLLeapHandListener* listener);
    void removeToolListener(SLLeapToolListener* listener);
    
    // @todo    can we declare this function private somehow
    virtual void onFrame(const Leap::Controller&);
    // unneeded rest of Leap::Listener interface, just kept here while developing
    // @todo remove the commented out functions below
    //virtual void onInit(const Leap::Controller&);
    //virtual void onConnect(const Leap::Controller&);
    //virtual void onDisconnect(const Leap::Controller&);
    //virtual void onExit(const Leap::Controller&);
    //virtual void onFocusGained(const Leap::Controller&);
    //virtual void onFocusLost(const Leap::Controller&);
    //virtual void onDeviceChange(const Leap::Controller&);
    //virtual void onServiceConnect(const Leap::Controller&);
    //virtual void onServiceDisconnect(const Leap::Controller&);

    const Leap::Controller& leapController() const { return _leapController; }
protected:
    Leap::Controller    _leapController;
    
    SLint               _prevFrameHandCount;
    SLint               _prevFrameToolCount;

    SLVLeapGestureListenerPtr   _gestureListeners;
    SLVLeapHandListenerPtr      _handListeners;
    SLVLeapToolListenerPtr      _toolListeners;
};


/*
We will have at most one leap controller in the world.
If someone want's to get data from it they have to register to 
a delegate (or we could implement it with an observer pattern)

We will support three types of delegates:
    1. Gestures
        > receives a custome SLLeapGesture object with the necessary data
    2. Hands
        > Receives a SLLeapHand object for left and right hand containing all hand and finger positions and orientations
    3. Tool
        > Receives a tool object containing tool relevant information, length, position, orientation.

*/



#endif