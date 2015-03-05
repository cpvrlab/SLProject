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

#include <stdafx.h>

using namespace std::chrono;
//-----------------------------------------------------------------------------
typedef std::chrono::high_resolution_clock              SLClock;
typedef std::chrono::high_resolution_clock::time_point  SLTimePoint;
//-----------------------------------------------------------------------------
//! High Resolution Timer class using C++11
/*!
High Resolution Timer that is able to measure the elapsed time with 1 
micro-second accuracy.
*/
class SLTimer
{
    public:
                        SLTimer();
                       ~SLTimer(){;}

            void        start();                   
            void        stop();
            SLfloat     getElapsedTimeInSec();
            SLfloat     getElapsedTimeInMilliSec();
            SLint64     getElapsedTimeInMicroSec();

    static  void        callAfterSleep(SLint milliSec, 
                                       function<void(void)> callbackFunc);
       
    private:
            SLTimePoint _timePoint1; //!< high precision start time point   
            SLTimePoint _timePoint2; //!< high precision end time point  
};
//---------------------------------------------------------------------------
#endif
