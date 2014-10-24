//#############################################################################
//  File:      SL/SLTimer.cpp
//  Author:    Marcus Hudritsch
//             Copied from: Song Ho Ahn (song.ahn@gmail.com), www.songho.ca
//  Purpose:   High Resolution SLTimer that is able to measure the elapsed time 
//             with 1 micro-second accuracy on Windows, Linux and Unix system
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: Song Ho Ahn (song.ahn@gmail.com)
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

//-----------------------------------------------------------------------------
SLTimer::SLTimer()
{
    #ifdef SL_OS_WINDOWS
    QueryPerformanceFrequency(&frequency);
    startCount.QuadPart = 0;
    endCount.QuadPart = 0;
    #else
    startCount.tv_sec = startCount.tv_usec = 0;
    endCount.tv_sec = endCount.tv_usec = 0;
    #endif

    stopped = 0;
    startTimeInMicroSec = 0;
    endTimeInMicroSec = 0;
}
//-----------------------------------------------------------------------------
SLTimer::~SLTimer()
{
}
//-----------------------------------------------------------------------------
// SLTimer::start starts timer. startCount will be set at this point.
void SLTimer::start()
{
    stopped = 0; // reset stop flag
    #ifdef SL_OS_WINDOWS
    QueryPerformanceCounter(&startCount);
    #else
    gettimeofday(&startCount, NULL);
    #endif
}
//-----------------------------------------------------------------------------
//! SLTimer::stop stops the timer. endCount will be set at this point.
void SLTimer::stop()
{
    stopped = 1; // set timer stopped flag

    #ifdef SL_OS_WINDOWS
    QueryPerformanceCounter(&endCount);
    #else
    gettimeofday(&endCount, NULL);
    #endif
}
//-----------------------------------------------------------------------------
/*! 
SLTimer::getElapsedTimeInMicroSec computes elapsed time in micro-second 
resolution. Other getElapsedTime will call this first, then convert to 
correspond resolution.
*/
double SLTimer::getElapsedTimeInMicroSec()
{
    #ifdef SL_OS_WINDOWS
    if(!stopped)
        QueryPerformanceCounter(&endCount);

    startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
    endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
    #else
    if(!stopped)
        gettimeofday(&endCount, NULL);

    startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
    endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
    #endif

    return endTimeInMicroSec - startTimeInMicroSec;
}
//-----------------------------------------------------------------------------
//! SLTimer::getElapsedTimeInMilliSec divides elapsedTimeInMicroSec by 1000
double SLTimer::getElapsedTimeInMilliSec()
{
    return this->getElapsedTimeInMicroSec() * 0.001;
}
//-----------------------------------------------------------------------------
//! SLTimer::getElapsedTimeInSec divide elapsedTimeInMicroSec by 1000000
double SLTimer::getElapsedTimeInSec()
{
    return this->getElapsedTimeInMicroSec() * 0.000001;
}
//-----------------------------------------------------------------------------

