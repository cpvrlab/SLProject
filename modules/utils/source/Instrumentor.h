//#############################################################################
//  File:      Utils/lib-utils/source/Instrumentor.h
//  Authors:   Cherno, adaptations by Marcus Hudritsch
//  Purpose:   Basic instrumentation profiler for writing performance measures
//             that can be used in Chrome://tracing app.
//  Original:  Based on https://gist.github.com/TheCherno
//  Changes:   Compared to Cherno's original I added in memory storage which is
//             faster and the addProfil function is now thread safe.
//  Date:      June 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//#############################################################################

#ifndef INSTRUMENTOR_H
#define INSTRUMENTOR_H

#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <thread>
#include <mutex>

/* Set PROFILING to 1 to enable profiling or to 0 for disabling profiling
 * Just add PROFILE_FUNCTION(); at the beginning of a function that you want to
 * profile. See the Instrumentor class below on how to display the profiling data.
 */
#define PROFILING 0

//-----------------------------------------------------------------------------
#if PROFILING
#    define BEGIN_PROFILING_SESSION(name, storeInMemory, outputPath) Instrumentor::get().beginSession(name, storeInMemory, outputPath)
#    define END_PROFILING_SESSION Instrumentor::get().endSession()
#    define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name)
#    define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else
#    define BEGIN_PROFILING_SESSION(name, storeInMemory, outputPath)
#    define END_PROFILING_SESSION
#    define PROFILE_SCOPE(name)
#    define PROFILE_FUNCTION()
#endif
//-----------------------------------------------------------------------------
struct ProfileResult
{
    const char* name;     //!< pointer to char has constant length
    long long   start;    //!< start time point
    long long   end;      //!< end time point
    uint32_t    threadID; //!< thread ID
};
//-----------------------------------------------------------------------------
struct InstrumentationSession
{
    std::string name;
};
//-----------------------------------------------------------------------------
//! Basic instrumentation profiler for Google Chrome tracing format
/*!
 Usage: include this header file somewhere in your code.
 In your most outer function (e.g. main) you have to begin profiling session with:
 Instrumentor::get().beginSession("Session Name"); If you pass storeInMemory=true
 the profileResults will be stored in memory instead of being written into the
 file stream which is pretty slow. Of course the in memory storage can quickly
 use a lot of memory depending how fine grained your profiling is.
 In app-Demo-SLProject this is done in SLInterface::slCreateAppAndScene.

 In between you can add either PROFILE_FUNCTION(); at the beginning of any routine
 or PROFILE_SCOPE(scopeName) at the beginning of any scope you want to measure.

 At the end of your most outer function (e.g. main) you end the session with:
 Instrumentor::get().endSession();

 After the endSession you can drag the Profiling-Results.json file into the
 chrome://tracing page of the Google Chrome browser.
 In app-Demo-SLProject this is done in SLInterface::slTerminate.
*/
class Instrumentor
{
public:
    Instrumentor() : _currentSession(nullptr), _profileCount(0) {}
    //.........................................................................
    void beginSession(const std::string& name,
                      const bool         storeInMemory = false,
                      const std::string& filePath      = "Profiling-Results.json")
    {
        _storeInMemory  = storeInMemory;
        _currentSession = new InstrumentationSession{name};
        _filePath       = filePath;

        if (!_storeInMemory)
        {
            _outputStream.open(_filePath);
            writeHeader();
        }
    }
    //.........................................................................
    void endSession()
    {
        if (_storeInMemory)
        {
            _outputStream.open(_filePath);
            writeHeader();

            for (auto result : _profileResults)
                writeProfile(result);
        }

        // end the file
        writeFooter();
        _outputStream.close();

        delete _currentSession;
        _currentSession = nullptr;
        _profileCount   = 0;
    }
    //.........................................................................
    /*! addProfile should be as fast as possible for not influencing the
        profiling by the profiler itself. In addition it must be thread safe.*/
    void addProfile(const ProfileResult& result)
    {
        std::lock_guard<std::mutex> lock(_mutex);

        if (_storeInMemory)
        {
            _profileResults.emplace_back(result);
        }
        else
        {
            writeProfile(result);
        }
    }
    //.........................................................................
    void writeProfile(const ProfileResult& result)
    {
        if (_profileCount++ > 0)
            _outputStream << ",";

        std::string name = result.name;
        std::replace(name.begin(), name.end(), '"', '\'');

        _outputStream << "{";
        _outputStream << "\"cat\":\"function\",";
        _outputStream << "\"dur\":" << (result.end - result.start) << ',';
        _outputStream << "\"name\":\"" << name << "\",";
        _outputStream << "\"ph\":\"X\",";
        _outputStream << "\"pid\":0,";
        _outputStream << "\"tid\":" << result.threadID << ",";
        _outputStream << "\"ts\":" << result.start;
        _outputStream << "}";

        // We constantly flush in case of file writing during profiling.
        if (!_storeInMemory)
            _outputStream.flush();
    }
    //.........................................................................
    void writeHeader()
    {
        _outputStream << "{\"otherData\": {},\"traceEvents\":[";
        _outputStream.flush();
    }
    //.........................................................................
    void writeFooter()
    {
        _outputStream << "]}";
        _outputStream.flush();
    }
    //.........................................................................
    static Instrumentor& get()
    {
        static Instrumentor instance;
        return instance;
    }
    //.........................................................................
    std::string filePath() { return _filePath; }
    //.........................................................................

private:
    InstrumentationSession*    _currentSession;
    std::string                _filePath;
    std::ofstream              _outputStream;
    int                        _profileCount;
    std::mutex                 _mutex;
    bool                       _storeInMemory = false;
    std::vector<ProfileResult> _profileResults;
};
//-----------------------------------------------------------------------------
typedef std::chrono::time_point<std::chrono::high_resolution_clock> HighResTimePoint;
//-----------------------------------------------------------------------------
class InstrumentationTimer
{
public:
    InstrumentationTimer(const char* name) : _name(name), _isStopped(false)
    {
        _startTimepoint = std::chrono::high_resolution_clock::now();
    }
    //.........................................................................
    ~InstrumentationTimer()
    {
        if (!_isStopped)
            stop();
    }
    //.........................................................................
    void stop()
    {
        auto endTimepoint = std::chrono::high_resolution_clock::now();

        long long start = std::chrono::time_point_cast<std::chrono::microseconds>(_startTimepoint).time_since_epoch().count();
        long long end   = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

        uint32_t threadID = (uint32_t)std::hash<std::thread::id>{}(std::this_thread::get_id());

        Instrumentor::get().addProfile({_name, start, end, threadID});

        _isStopped = true;
    }
    //.........................................................................

private:
    const char*      _name;           //!< function or scope name as char pointer (not std::string)
    HighResTimePoint _startTimepoint; //!< start timepoint
    bool             _isStopped;      //!< flag if timer got stopped
};
//-----------------------------------------------------------------------------
#endif INSTRUMENTER_H