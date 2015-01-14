#include  <stdafx.h>
#include <SLLeapController.h>

#include <set>


/** Constructor */
SLLeapController::SLLeapController()
{
    _leapController.addListener(*this);
    _leapController.setPolicy(Leap::Controller::POLICY_BACKGROUND_FRAMES);
}

SLLeapController::~SLLeapController()
{
    _leapController.removeListener(*this);
}


void SLLeapController::onFrame(const Leap::Controller& controller)
{
    const Leap::Frame frame = _leapController.frame();


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


    // @todo    notify tool listeners

    // @todo    notify gesture listeners
}


void SLLeapController::registerGestureListener(SLLeapGestureListener* listener)
{
    _gestureListeners.push_back(listener);
}
void SLLeapController::registerHandListener(SLLeapHandListener* listener)
{
    _handListeners.push_back(listener);
}
void SLLeapController::registerGestureListener(SLLeapToolListener* listener)
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

void SLLeapController::removeGestureListener(SLLeapToolListener* listener)
{
    SLVLeapToolListenerPtr& v = _toolListeners;
    v.erase(remove(v.begin(), v.end(), listener), v.end());
}