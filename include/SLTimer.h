//#############################################################################
//  File:      SL/SLTimer.h
//  Author:    Marcus Hudritsch
//             Copied from: Song Ho Ahn (song.ahn@gmail.com), www.songho.ca
//  Purpose:   High Resolution Timer that is able to measure the elapsed time 
//             with 1 micro-second accuracy on Windows, Linux and Unix system
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: Song Ho Ahn (song.ahn@gmail.com)
//#############################################################################

#ifndef SLTIMER
#define SLTIMER

#include <stdafx.h>

//-----------------------------------------------------------------------------
//! High Resolution Timer class
/*!
High Resolution Timer that is able to measure the elapsed time with 1 
micro-second accuracy on Windows, Linux and Unix system
*/
class SLTimer
{
    public:
                    SLTimer();
                    ~SLTimer();

            void     start();                   
            void     stop();
            double   getElapsedTimeInSec();
            double   getElapsedTimeInMilliSec();
            double   getElapsedTimeInMicroSec();
       
    private:
            double   startTimeInMicroSec;       // starting time in micro-second
            double   endTimeInMicroSec;         // ending time in micro-second
            int      stopped;                   // stop flag

            #ifdef SL_OS_WINDOWS
            LARGE_INTEGER frequency;            //!< ticks per second
            LARGE_INTEGER startCount;           //!< ticks at start
            LARGE_INTEGER endCount;             //!< ticks at end
            #else
            timeval startCount;
            timeval endCount;
            #endif
};
//---------------------------------------------------------------------------
#endif
