//#############################################################################
//  File:      SLLeapController.cpp
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include  <stdafx.h>
#include <SLLeapController.h>

#include <set>


/** Constructor */
SLLeapController::SLLeapController()
{
    _leapController.setPolicy(Leap::Controller::POLICY_BACKGROUND_FRAMES);
    _leapController.enableGesture(Leap::Gesture::TYPE_SWIPE);
    _leapController.enableGesture(Leap::Gesture::TYPE_KEY_TAP);
    _leapController.enableGesture(Leap::Gesture::TYPE_SCREEN_TAP);
    _leapController.enableGesture(Leap::Gesture::TYPE_SWIPE);
}

SLLeapController::~SLLeapController()
{
}

void SLLeapController::poll()
{
    const Leap::Frame frame = _leapController.frame();

    if (_prevFrameId != frame.id())
        onFrame(frame);

    _prevFrameId = frame.id();
}

void SLLeapController::onFrame(const Leap::Frame frame)
{
    if (_handListeners.size())
    {
        std::vector<SLLeapHand> slHands;
             
        Leap::HandList hands = frame.hands();
        for (Leap::HandList::const_iterator hl = hands.begin(); hl != hands.end(); ++hl)
        {
            const Leap::Hand hand = *hl;

            if (hand.isValid()) {
                SLLeapHand slHand;
                slHand.leapHand(hand);
                slHands.push_back(slHand);
            }
        }

        if (slHands.size() || _prevFrameHandCount > 0)
        {
            for (SLint i = 0; i < _handListeners.size(); ++i)
                _handListeners[i]->onLeapHandChange(slHands);
        }

        _prevFrameHandCount = hands.count();
    }

    // notify tool listeners
    if (_toolListeners.size())
    {
        std::vector<SLLeapTool> slTools;
             
        Leap::ToolList tools = frame.tools();
        for (Leap::ToolList::const_iterator tl = tools.begin(); tl != tools.end(); ++tl)
        {
            const Leap::Tool tool = *tl;

            if (tool.isValid()) {
                SLLeapTool slTool;
                slTool.leapTool(tool);
                slTools.push_back(slTool);
            }
        }

        if (slTools.size() || _prevFrameToolCount > 0)
        {
            for (SLint i = 0; i < _toolListeners.size(); ++i)
                _toolListeners[i]->onLeapToolChange(slTools);
        }

        _prevFrameToolCount = tools.count();
    }

    // notify gesture listeners
    if (_gestureListeners.size())
    {
        Leap::GestureList gestures = frame.gestures();
        for (Leap::GestureList::const_iterator gl = gestures.begin(); gl != gestures.end(); ++gl)
        {
            Leap::Gesture gesture = *gl;

            SLLeapGesture* slGesture = NULL;
            switch (gesture.type())
            {
            case Leap::Gesture::TYPE_CIRCLE:       slGesture = new SLLeapCircleGesture; break;
            case Leap::Gesture::TYPE_SWIPE:        slGesture = new SLLeapSwipeGesture; break;
            case Leap::Gesture::TYPE_KEY_TAP:      slGesture = new SLLeapKeyTapGesture; break;
            case Leap::Gesture::TYPE_SCREEN_TAP:   slGesture = new SLLeapScreenTapGesture; break;
            }

            if (slGesture)
            {
                slGesture->leapGesture(gesture);
                for (SLint i = 0; i < _gestureListeners.size(); ++i)
                    _gestureListeners[i]->onLeapGesture(*slGesture);

                delete slGesture;
            }
        }
    }
}


void SLLeapController::registerGestureListener(SLLeapGestureListener* listener)
{
    _gestureListeners.push_back(listener);
}
void SLLeapController::registerHandListener(SLLeapHandListener* listener)
{
    _handListeners.push_back(listener);
}
void SLLeapController::registerToolListener(SLLeapToolListener* listener)
{
    _toolListeners.push_back(listener);
}
    
void SLLeapController::removeGestureListener(SLLeapGestureListener* listener)
{    
    SLVLeapGestureListenerPtr& v = _gestureListeners;
    v.erase(remove(v.begin(), v.end(), listener), v.end());
}
void SLLeapController::removeHandListener(SLLeapHandListener* listener)
{
    SLVLeapHandListenerPtr& v = _handListeners;
    v.erase(remove(v.begin(), v.end(), listener), v.end());
}

void SLLeapController::removeToolListener(SLLeapToolListener* listener)
{
    SLVLeapToolListenerPtr& v = _toolListeners;
    v.erase(remove(v.begin(), v.end(), listener), v.end());
}