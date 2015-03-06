//#############################################################################
//  File:      SL/SLTimer.cpp
//  Author:    Marcus Hudritsch
//             Copied from: Song Ho Ahn (song.ahn@gmail.com), www.songho.ca
//  Purpose:   High Resolution SLTimer that is able to measure the elapsed time 
//             with 1 micro-second accuracy on Windows, Linux and Unix system
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Song Ho Ahn (song.ahn@gmail.com)
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

using namespace std::placeholders;

//-----------------------------------------------------------------------------
SLTimer::SLTimer()
{
}
//-----------------------------------------------------------------------------
// SLTimer::start starts timer. startCount will be set at this point.
void SLTimer::start()
{
    _timePoint1 = SLClock::now();
}
//-----------------------------------------------------------------------------
//! SLTimer::stop stops the timer. endCount will be set at this point.
void SLTimer::stop()
{
    _timePoint2 = SLClock::now();
}
//-----------------------------------------------------------------------------
/*! 
SLTimer::getElapsedTimeInMicroSec computes elapsed time in micro-second 
resolution. Other getElapsedTime will call this first, then convert to 
correspond resolution.
*/
inline SLint64 SLTimer::getElapsedTimeInMicroSec()
{
    return duration_cast<microseconds>(SLClock::now()-_timePoint1).count();
}
//-----------------------------------------------------------------------------
//! SLTimer::getElapsedTimeInMilliSec divides elapsedTimeInMicroSec by 1000
SLfloat SLTimer::getElapsedTimeInMilliSec()
{
    return getElapsedTimeInMicroSec() * 0.001f;
}
//-----------------------------------------------------------------------------
//! SLTimer::getElapsedTimeInSec divide elapsedTimeInMicroSec by 1000000
SLfloat SLTimer::getElapsedTimeInSec()
{
    return getElapsedTimeInMicroSec() * 0.000001f;
}
//-----------------------------------------------------------------------------
//! Delayed call of the callback function after the passed milliseconds.
void SLTimer::callAfterSleep(SLint milliSec, function<void(void)> callbackFunc)
{
    // Create a thread that immediatelly sleeps the milliseconds
    thread t
    (   [=]()
        {   this_thread::sleep_for(chrono::milliseconds(milliSec)); 
            callbackFunc();
        }
    );

    // detach the thread so that it can exist after the block
    t.detach();
}
//-----------------------------------------------------------------------------


