//#############################################################################
//  File:      Profiler.h
//  Authors:   Marino von Wattenwyl
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <fstream>

//-----------------------------------------------------------------------------
/* Set PROFILING to 1 to enable profiling or to 0 for disabling profiling
 * Just add PROFILE_FUNCTION(); at the beginning of a function that you want to
 * profile. See the Profiler class below on how to display the profiling data.
 */
#define PROFILING 0
//-----------------------------------------------------------------------------
#if PROFILING
#    define BEGIN_PROFILING_SESSION(filePath) Profiler::instance().beginSession(filePath)
#    define PROFILE_SCOPE(name) ProfilerTimer profilerTimer##__LINE__(name)
#    define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#    define PROFILE_THREAD(name) Profiler::instance().profileThread(name)
#    define PROFILER_TRACE_FILE_PATH Profiler::instance().filePath()
#    define END_PROFILING_SESSION() Profiler::instance().endSession()
#else
#    define BEGIN_PROFILING_SESSION(filePath)
#    define PROFILE_SCOPE(name)
#    define PROFILE_FUNCTION()
#    define PROFILE_THREAD(name)
#    define PROFILER_TRACE_FILE_PATH
#    define END_PROFILING_SESSION()
#endif
//-----------------------------------------------------------------------------
struct ProfilingResult
{
    const char* name;     //!< Name of the profiled function/scope
    uint32_t    depth;    //!< Depth of the scope in it's thread's call stack
    uint64_t    start;    //!< Start time in microseconds relative to the session start
    uint64_t    end;      //!< End time in microseconds relative to the session start
    uint32_t    threadId; //!< ID of the thread in which the scope was entered
};
//-----------------------------------------------------------------------------
//! Utility for profiling functions/scopes and writing the results to a file.
/*!
 * To start the profiling, call BEGIN_PROFILING_SESSION(filePath) with the path
 * to the trace file. After that you can place "PROFILE_FUNCTION();" or
 * "PROFILE_SCOPE(name);" at the start of every function or scope you want to
 * measure.
 * The profiler supports multithreading. To add a new thread, call
 * "PROFILE_THREAD(name)" at the start of the thread. Threads with the same
 * name will appear merged in the trace file. To end the session and write
 * the result to the trace file, call END_PROFILING_SESSION().
 *
 * The resulting trace gets written into the data folder of SLProject and can
 * be opened using the trace viewer located at /externals/trace-viewer/trace-viewer.jar.
 * Note that a Java Runtime Environment is required to launch this JAR archive.
 */
class Profiler
{
public:
    static Profiler& instance()
    {
        static Profiler instance;
        return instance;
    }

    void        beginSession(std::string filePath);
    std::string filePath() { return _filePath; }
    void        endSession();

    void recordResult(ProfilingResult result);
    void profileThread(const std::string& name);

private:
    static void writeString(const char* s, std::ofstream& stream);

private:
    std::string                  _filePath;         //!< Future path of the trace file
    uint64_t                     _sessionStart = 0; //!< Start timestamp of the session in microseconds
    std::vector<ProfilingResult> _results;          //!< List of profiling results (of all threads)
    std::vector<std::string>     _threadNames;      //!< List of thread names (the thread ID is the index)
    std::mutex                   _mutex;            //!< Mutex for accessing profiling results and thread names
};
//-----------------------------------------------------------------------------
//! A timer for profiling functions and scopes
/*!
 * This class should be instantiated at the start of functions and scopes that
 * should be profiled. The object will record the current time at it's construction
 * and the current time at it's destruction (when the scope ends) and the depth
 * in the call stack. The destructor automatically calls Profiler::instance().recordResult().
 */
class ProfilerTimer
{
    friend class Profiler;

public:
    explicit ProfilerTimer(const char* name);
    ~ProfilerTimer();

private:
    static constexpr uint32_t    INVALID_THREAD_ID = -1;
    static thread_local uint32_t threadId;
    static thread_local uint32_t threadDepth;

    const char*                                                 _name;
    uint32_t                                                    _depth;
    std::chrono::time_point<std::chrono::high_resolution_clock> _startPoint;
    bool                                                        _running;
};
//-----------------------------------------------------------------------------
#endif // PROFILER_H
