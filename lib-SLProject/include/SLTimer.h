//#############################################################################
//  File:      SL/SLTimer.h
//  Author:    Marcus Hudritsch
//  Purpose:   High Resolution Timer that is able to measure the elapsed time
//             with 1 micro-second accuracy with C++11 high_resolution_clock
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Song Ho Ahn (song.ahn@gmail.com)
//#############################################################################

#ifndef SLTIMER
#define SLTIMER

#include <SL.h>

using namespace std::chrono;

//! High Resolution Timer class using C++11
/*!
High Resolution Timer that is able to measure the elapsed time with 1 
micro-second accuracy.
*/
class SLTimer
{
    public:
    SLTimer() { _timePoint1 = SLClock::now(); }
    ~SLTimer() { ; }

    void    start() { _timePoint1 = SLClock::now(); }
    void    stop() { _timePoint2 = SLClock::now(); }
    SLfloat elapsedTimeInSec() { return duration_cast<seconds>(SLClock::now() - _timePoint1).count(); }
    SLfloat elapsedTimeInMilliSec() { return duration_cast<milliseconds>(SLClock::now() - _timePoint1).count(); }
    SLint64 elapsedTimeInMicroSec() { return duration_cast<microseconds>(SLClock::now() - _timePoint1).count(); }

    static void callAfterSleep(SLint                milliSec,
                               function<void(void)> callbackFunc);

    private:
    SLTimePoint _timePoint1; //!< high precision start time point
    SLTimePoint _timePoint2; //!< high precision end time point
};
//---------------------------------------------------------------------------
#endif
