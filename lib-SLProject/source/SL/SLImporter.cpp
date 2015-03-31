//#############################################################################
//  File:      SL/SLImporter.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <cstdarg> // only needed because we wrap pintf in logMessage, read the todo and fix it!
#include <SLImporter.h>

//-----------------------------------------------------------------------------
//! Default path for 3DS models used when only filename is passed in load.
SLstring SLImporter::defaultPath = "../_data/models/";

//-----------------------------------------------------------------------------
/*! Default constructor, doesn't log anything
*/
SLImporter::SLImporter()
           : _logConsoleVerbosity(LV_Quiet),
             _logFileVerbosity(LV_Quiet),
             _sceneRoot(nullptr),
             _skeleton(nullptr)
{ }

//-----------------------------------------------------------------------------
/*! Constructor that only outputs console logs
*/
SLImporter::SLImporter(SLLogVerbosity consoleVerb)
           : _logFileVerbosity(LV_Quiet),
            _sceneRoot(nullptr),
            _skeleton(nullptr)
{ }

//-----------------------------------------------------------------------------
/*! Constructor that allows logging to a file with different verbosity
*/
SLImporter::SLImporter(const SLstring& logFile, SLLogVerbosity logConsoleVerb, SLLogVerbosity logFileVerb)
           : _logConsoleVerbosity(logConsoleVerb),
             _logFileVerbosity(logFileVerb),
             _sceneRoot(nullptr),
             _skeleton(nullptr)
{ 
    if (_logFileVerbosity > LV_Quiet)
        _log.open(logFile.c_str());
}

//-----------------------------------------------------------------------------
/*! Destructor, closes the file stream if it was used
*/
SLImporter::~SLImporter()
{
    if (_log.is_open())
        _log.close();
}

//-----------------------------------------------------------------------------
/*! Logs messages to the importer logfile and the console
    @param  msg          the message to add to the log
    @param  verbosity    the verbosity of the message
    @todo   Build a dedicated log class that can be instantiated (so the importer can have its own)
            Let this log class write to file etc.
            Don't use printf anymore, its c. (c++11 has to_str, else we have to work with ss (ugh...))
            I only used printf here because it allows me to combine a string with different variables
            in only one line and I don't have an easy way to do this in c++0x. Again c++11 would be easy.
*/
void SLImporter::logMessage(SLLogVerbosity verbosity, const char* msg, ...)
{
    #if defined(SL_OS_ANDROID)
    #define SL_LOG(msg);
    #else
    // write message to a buffer
    char buffer[4096];
    std::va_list arg;
    va_start(arg, msg);
    std::vsnprintf(buffer, 4096, msg, arg);
    va_end(arg);

    if (_logConsoleVerbosity >= verbosity)
        SL_LOG("%s", buffer);
    if (_logFileVerbosity >= verbosity)
    {
        _log << buffer;
        _log.flush();
    }
    #endif
}
//-----------------------------------------------------------------------------
