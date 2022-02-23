//#############################################################################
//  File:      Utils/lib-utils/source/HighResTimer.h
//  Authors:   Marcus Hudritsch
//  Purpose:   High Resolution Timer that is able to measure the elapsed time
//             with 1 micro-second accuracy with C++11 high_resolution_clock
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Song Ho Ahn (song.ahn@gmail.com)
//#############################################################################

#ifndef HIGHRESTIMER
#define HIGHRESTIMER

#include <chrono>
#include <functional>
#include <thread>

using namespace std;
using namespace std::chrono;

typedef std::chrono::high_resolution_clock             HighResClock;
typedef std::chrono::high_resolution_clock::time_point HighResTimePoint;

//! High Resolution Timer class using C++11
/*!
High Resolution Timer that is able to measure the elapsed time with 1
micro-second accuracy.
*/
class HighResTimer
{
public:
    HighResTimer() { _timePoint1 = HighResClock::now(); }

    void    start() { _timePoint1 = HighResClock::now(); }
    void    stop() { _timePoint2 = HighResClock::now(); }
    float   elapsedTimeInSec() { return (float)((double)elapsedTimeInMicroSec() / 1000000.0); }
    float   elapsedTimeInMilliSec() { return (float)((double)elapsedTimeInMicroSec() / 1000.0); }
    int64_t elapsedTimeInMicroSec() { return duration_cast<microseconds>(HighResClock::now() - _timePoint1).count(); }

    static void callAfterSleep(int milliSec, const function<void(void)>& callbackFunc)
    {
        // Create a thread that immediately sleeps the milliseconds
        thread t([=]()
                 {
            this_thread::sleep_for(chrono::milliseconds(milliSec));
            callbackFunc(); });

        // detach the thread so that it can exist after the block
        t.detach();
    }

private:
    HighResTimePoint _timePoint1; //!< high precision start time point
    HighResTimePoint _timePoint2; //!< high precision end time point
};
//---------------------------------------------------------------------------
#endif
