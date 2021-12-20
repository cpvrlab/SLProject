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
#define PROFILING 0
//-----------------------------------------------------------------------------
#ifdef PROFILING
#    define BEGIN_PROFILING_SESSION(filePath) Profiler::instance().beginSession(filePath)
#    define PROFILE_SCOPE(name) ProfilerTimer profilerTimer##__LINE__(name)
#    define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#    define NAME_PROFILED_THREAD(name) Profiler::instance().nameCurrentThread(name)
#    define PROFILER_TRACE_FILE_PATH Profiler::instance().filePath()
#    define END_PROFILING_SESSION() Profiler::instance().endSession()
#else
#    define BEGIN_PROFILING_SESSION(filePath)
#    define PROFILE_SCOPE(name)
#    define PROFILE_FUNCTION()
#    define NAME_PROFILED_THREAD(name)
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
    uint32_t    threadId; //!< Hash of the thread ID
};
//-----------------------------------------------------------------------------
//! Utility class for profiling functions/scopes and writing the results to a file.
/*!
 * To start the profiling, call Profiler::instance().beginSession(filePath) with the path to the trace file.
 * After that you can place "PROFILE_FUNCTION();" or "PROFILE_SCOPE(name);" at the start of
 * every function or scope you want to measure.
 * The profiler supports multithreading. The current thread can be named with "NAME_PROFILED_THREAD(name);".
 * If no thread name is specified at the end of the session, the name will be "Thread#ID".
 * To end the session and write the result to the trace file, call Profiler::instance().endSession().
 *
 * The resulting trace file can be opened using the trace viewer located at /externals/trace-viewer/trace-viewer.jar.
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
    void nameCurrentThread(const std::string& name);

private:
    static void writeInt32(std::ofstream& stream, uint32_t i);
    static void writeInt64(std::ofstream& stream, uint64_t i);
    static void writeString(std::ofstream& stream, const char* s);
    static bool isLittleEndian();

private:
    std::string                     _filePath;         //!< Future path of the trace file
    uint64_t                        _sessionStart = 0; //!< Start timestamp of the session in microseconds
    std::vector<ProfilingResult>    _results;          //!< List of profiling results (of all threads)
    std::map<uint32_t, std::string> _threadNames;      //!< Map from thread ID hashes to thread names
    std::mutex                      _mutex;            //!< Mutex for accessing profiling results and thread names
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
public:
    explicit ProfilerTimer(const char* name);
    ~ProfilerTimer();

private:
    thread_local static int threadDepth;

    const char*                                                 _name;
    uint32_t                                                    _depth;
    std::chrono::time_point<std::chrono::high_resolution_clock> _startPoint;
    bool                                                        _running;
};
//-----------------------------------------------------------------------------
#endif // PROFILER_H
