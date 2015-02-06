//#############################################################################
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
#include "SLLeapFinger.h"
#include "SLLeapHand.h"
#include "SLLeapTool.h"
#include "SLLeapGesture.h"

#include <SLInputDevice.h>


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

// leap controller input device
class SLLeapController : public SLInputDevice, public Leap::Listener
{
public:
                    SLLeapController();
                    ~SLLeapController();
        
            void    registerGestureListener(SLLeapGestureListener* listener);
            void    registerHandListener   (SLLeapHandListener* listener);
            void    registerToolListener   (SLLeapToolListener* listener);
    
            void    removeGestureListener  (SLLeapGestureListener* listener);
            void    removeHandListener     (SLLeapHandListener* listener);
            void    removeToolListener     (SLLeapToolListener* listener);
    
    // SLInputDevice function implementation
    virtual SLbool poll();

    virtual void onFrame(const Leap::Controller&)
    {
        SLint i = 0;
    }
    
protected:
    int64_t                     _prevFrameId;
    Leap::Controller            _leapController;
    
    SLint                       _prevFrameHandCount;
    SLint                       _prevFrameToolCount;

    SLVLeapGestureListenerPtr   _gestureListeners;
    SLVLeapHandListenerPtr      _handListeners;
    SLVLeapToolListenerPtr      _toolListeners;

    
    virtual void onFrame(const Leap::Frame&);
};


#endif
