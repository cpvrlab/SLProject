//#############################################################################
//  File:      SLCVTrackingStateMachine.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCV_TRACKINGSTATEMACHINE_H
#define SLCV_TRACKINGSTATEMACHINE_H

class SLCVMapTracking;

/*!State machine for SLCVMapTracking state. It runs in the same thread as the tracking.
*/
class SLCVTrackingStateMachine
{
public:
    enum TrackingState
    {
        INITIALIZING = 0,
        IDLE,
        TRACKING_LOST,
        TRACKING_OK
    };

    //!ctor
    SLCVTrackingStateMachine(SLCVMapTracking* tracking);

    std::string getPrintableState();

    //!getter for current state
    TrackingState state() { return mState; }
    //!getter for last state
    TrackingState lastProcessedState() { return mLastProcessedState; }

    //!request state idle
    void requestStateIdle();
    //!If system is in idle, it resumes depending on if system is initialized
    void requestResume();
    //!check current state
    bool hasStateIdle();

    //!state transition check and apply
    void stateTransition();

private:
    TrackingState mState = IDLE;
    TrackingState mLastProcessedState = IDLE;

    //!reset external requests
    void resetRequests();

    //!System switches to IDLE state as soon as possible.
    bool _resumeRequested = false;
    //!System switches to IDLE state as soon as possible.
    bool _idleRequested = false;
    //!Tracking is working and requests state tracking ok
    bool _trackingOKRequested = false;
    //!Tracking is not working and equests state tracking lost
    bool _trackingLostRequested = false;

    //pointer to tracking instance that uses this state machine
    SLCVMapTracking* _tracking = NULL;

    std::mutex _mutexStates;
    //serial usage
    bool _serial = true;
};

#endif //SLCV_TRACKINGSTATEMACHINE_H


