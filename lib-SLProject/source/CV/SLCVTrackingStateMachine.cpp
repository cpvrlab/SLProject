//#############################################################################
//  File:      SLCVTrackingStateMachine.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCVTrackingStateMachine.h>
#include <SLCVMapTracking.h>

//-----------------------------------------------------------------------------
SLCVTrackingStateMachine::SLCVTrackingStateMachine(SLCVMapTracking* tracking)
    : _tracking(tracking)
{
}
//-----------------------------------------------------------------------------
string SLCVTrackingStateMachine::getPrintableState()
{
    switch (mState)
    {
    case RESETTING:
        return "RESETTING";
    case INITIALIZING:
        return "INITIALIZING";
    case IDLE:
        return "IDLE";
    case TRACKING_LOST:
        return "TRACKING_LOST"; //motion model tracking
    case TRACKING_OK:
        return "TRACKING_OK";

        return "";
    }
}
//-----------------------------------------------------------------------------
//!request state idle
void SLCVTrackingStateMachine::requestStateIdle()
{
    std::lock_guard<std::mutex> guard(_mutexStates);
    _idleRequested = true;
}
//-----------------------------------------------------------------------------
//!If system is in idle, it resumes with INITIAIZED or NOT_INITIALIZED state depending on if system is initialized.
void SLCVTrackingStateMachine::requestResume()
{
    std::lock_guard<std::mutex> guard(_mutexStates);
    _resumeRequested = true;
}
//-----------------------------------------------------------------------------
//!request reset. state switches to idle afterwards.
void SLCVTrackingStateMachine::requestReset()
{
    std::lock_guard<std::mutex> guard(_mutexStates);
    _resetRequested = true;
}
//-----------------------------------------------------------------------------
//!check current state
bool SLCVTrackingStateMachine::hasStateIdle()
{
    std::lock_guard<std::mutex> guard(_mutexStates);
    return mState == IDLE;
}
//-----------------------------------------------------------------------------
void SLCVTrackingStateMachine::stateTransition()
{
    std::lock_guard<std::mutex> guard(_mutexStates);

    //store last state
    mLastProcessedState = mState;

    //requests for reset and idle beat all other state transitions
    if (_resetRequested)
    {
        mState = RESETTING;
    }
    else if (_idleRequested)
    {
        mState = IDLE;
    }

    if (mState == RESETTING)
    {
        //we switch directly to idle not initialized state
        mState = IDLE;
    }
    else if (mState == IDLE)
    {
        if (_resumeRequested)
        {
            if (!_tracking->isInitialized())
            {
                mState = INITIALIZING;
            }
            else
            {
                mState = TRACKING_LOST;
            }
        }
    }
    else if (mState == INITIALIZING)
    {
        if (_tracking->isInitialized())
        {
            if (_tracking->isOK())
            {
                mState = TRACKING_OK;
            }
            else
            {
                mState = TRACKING_LOST;
            }
        }
    }
    else if (mState == TRACKING_OK)
    {
        if (!_tracking->isOK())
            mState = TRACKING_LOST;
    }
    else if (mState == TRACKING_LOST)
    {
        if (_tracking->isOK())
            mState = TRACKING_OK;
    }

    //reset all old requests
    resetRequests();
}
//-----------------------------------------------------------------------------
void SLCVTrackingStateMachine::resetRequests()
{
    _idleRequested = false;
    _resetRequested = false;
    _resumeRequested = false;
    _trackingOKRequested = false;
    _trackingLostRequested = false;
}
//-----------------------------------------------------------------------------