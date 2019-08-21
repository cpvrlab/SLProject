//#############################################################################
//  File:      SL/SLTimer.cpp
//  Author:    Marcus Hudritsch
//             Copied from: Song Ho Ahn (song.ahn@gmail.com), www.songho.ca
//  Purpose:   High Resolution SLTimer that is able to measure the elapsed time
//             with 1 micro-second accuracy on Windows, Linux and Unix system
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Song Ho Ahn (song.ahn@gmail.com)
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

using namespace std::placeholders;

//-----------------------------------------------------------------------------
//! Delayed call of the callback function after the passed milliseconds.
void SLTimer::callAfterSleep(SLint milliSec, function<void(void)> callbackFunc)
{
    // Create a thread that immediatelly sleeps the milliseconds
    thread t([=]() {
        this_thread::sleep_for(chrono::milliseconds(milliSec));
        callbackFunc();
    });

    // detach the thread so that it can exist after the block
    t.detach();
}
//-----------------------------------------------------------------------------
