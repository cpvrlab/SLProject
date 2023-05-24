//#############################################################################
//  File:      Profiler.cpp
//  Authors:   Marino von Wattenwyl
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <Profiler.h>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ByteOrder.h>

//-----------------------------------------------------------------------------
/*!
 * Starts a profiling session by saving the session start timestamp so it
 * can later be subtracted from the individual result timestamps to get the
 * time points relative to the start of the session.
 * @param filePath The path where the trace file should be written to
 */
void Profiler::beginSession(std::string filePath)
{
    _filePath = std::move(filePath);

    auto startPoint = std::chrono::high_resolution_clock::now();
    _sessionStart   = std::chrono::time_point_cast<std::chrono::microseconds>(startPoint).time_since_epoch().count();
}
//-----------------------------------------------------------------------------
/*! Ends the profiling session and writes the result to a trace file.
 * A trace file (.slt) has the following layout:
 * Number of scopes: int32
 *  Scope 1 name: (length: int32, name: non-null-terminated char array)
 *  Scope 2 name: (length: int32, name: non-null-terminated char array)
 *  ...
 * Number of threads: int32
 *  Thread 1 name: (length: int32, name: non-null-terminated char array)
 *   Number of scopes entered in thread 1: int32
 *    Scope 1 in thread 1 (name index: int32, depth: int32, start time: int64, end time: int64)
 *    Scope 2 in thread 1 (name index: int32, depth: int32, start time: int64, end time: int64)
 *    ...
 *  Thread 2 name: (length: int32, name: non-null-terminated char array)
 *   Number of scopes entered in thread 2: int32
 *    Scope 1 in thread 2 (name index: int32, depth: int32, start time: int64, end time: int64)
 *    Scope 2 in thread 2 (name index: int32, depth: int32, start time: int64, end time: int64)
 *    ...
 *  ...
 *
 *  All data is written in the big-endian format because that's the endianness that
 *  Java uses to read the data later on in the trace viewer.
 *  This means that the function has to check the endianness of the system
 *  and convert all integers to big endian if we're on a little-endian system.
 */
void Profiler::endSession()
{
    std::ofstream fileStream(_filePath, std::ios::binary);

    ////////////////////////////////////////
    // Collect scope names and thread IDs //
    ////////////////////////////////////////

    std::vector<const char*> scopeNames;
    std::vector<uint32_t>    threadIds;

    for (ProfilingResult& result : _results)
    {
        if (std::find(scopeNames.begin(), scopeNames.end(), result.name) == scopeNames.end())
            scopeNames.push_back(result.name);

        if (std::find(threadIds.begin(), threadIds.end(), result.threadId) == threadIds.end())
            threadIds.push_back(result.threadId);
    }

    /////////////////////////
    // Write scope section //
    /////////////////////////

    // Write the number of scope names
    auto numScopeNames = (uint32_t)scopeNames.size();
    ByteOrder::writeBigEndian32(numScopeNames, fileStream);

    // Write each scope name
    for (const char* scopeName : scopeNames)
    {
        writeString(scopeName, fileStream);
    }

    /////////////////////////
    // Write trace section //
    /////////////////////////

    // Write number of threads
    auto numThreads = (uint32_t)threadIds.size();
    ByteOrder::writeBigEndian32(numThreads, fileStream);

    for (uint32_t threadId : threadIds)
    {
        // Write thread name
        writeString(_threadNames[threadId].c_str(), fileStream);

        // Count and write number of scopes in thread
        uint32_t numScopes = 0;
        for (ProfilingResult& result : _results)
        {
            if (result.threadId == threadId) numScopes++;
        }
        ByteOrder::writeBigEndian32(numScopes, fileStream);

        // Write results of thread
        for (ProfilingResult& result : _results)
        {
            if (result.threadId != threadId) continue;

            auto nameIndex = (uint32_t)(std::find(scopeNames.begin(), scopeNames.end(), result.name) - scopeNames.begin());
            auto depth     = result.depth;
            auto start     = result.start - _sessionStart;
            auto end       = result.end - _sessionStart;

            ByteOrder::writeBigEndian32(nameIndex, fileStream);
            ByteOrder::writeBigEndian32(depth, fileStream);
            ByteOrder::writeBigEndian64(start, fileStream);
            ByteOrder::writeBigEndian64(end, fileStream);
        }
    }
}
//-----------------------------------------------------------------------------
/*!
 * Stores a result thread-safely in a vector so it can be written to a trace
 * file at the end of the session.
 * @param result
 */
void Profiler::recordResult(ProfilingResult result)
{
    _mutex.lock();
    _results.push_back(result);
    _mutex.unlock();
}
//-----------------------------------------------------------------------------
/*!
 * Associates the thread in which the function was called with the name provided.
 * This function must be called at the start of every profiled thread.
 * It is sensibly also thread-safe.
 * @param name
 */
void Profiler::profileThread(const std::string& name)
{
    _mutex.lock();

    for (uint32_t i = 0; i < _threadNames.size(); i++)
    {
        if (_threadNames[i] == name)
        {
            ProfilerTimer::threadId = i;
            _mutex.unlock();
            return;
        }
    }

    uint32_t threadId = (uint32_t)_threadNames.size();
    _threadNames.push_back(name);
    ProfilerTimer::threadId = threadId;

    _mutex.unlock();
}
//-----------------------------------------------------------------------------
//! Writes the length (32-bit) and the string (non-null-terminated) itself to the file stream
void Profiler::writeString(const char* s, std::ofstream& stream)
{
    ByteOrder::writeBigEndian32((uint32_t)std::strlen(s), stream);
    stream << s;
}
//-----------------------------------------------------------------------------
thread_local uint32_t ProfilerTimer::threadId    = INVALID_THREAD_ID;
thread_local uint32_t ProfilerTimer::threadDepth = 0;
//-----------------------------------------------------------------------------
/*!
 * Constructor for ProfilerTimer that saves the current time as the start
 * time, the thread-local depth as the scope depth and increases the
 * thread-local depth since we have just entered a scope.
 * PROFILE_THREAD must be called in the current thread before this function
 * or else the current thread can't be identified and the application exits.
 * @param name Name of the scope
 */
ProfilerTimer::ProfilerTimer(const char* name)
{
    // If the thread ID is INVALID_THREAD_ID, PROFILE_THREAD hasn't been called
    // We don't know the current thread in this case, so we simply skip
    if (threadId == INVALID_THREAD_ID)
    {
        _running = false;
        std::cout << ("Warning: Attempted to profile scope in non-profiled thread\nScope name: " + std::string(name) + "\n").c_str();
        return;
    }

    _name       = name;
    _startPoint = std::chrono::high_resolution_clock::now();
    _depth      = threadDepth;
    _running    = true;

    threadDepth++;
}
//-----------------------------------------------------------------------------
/*!
 * Destructor for ProfilerTimer that creates a ProfilingResult with
 * the scope name, the depth, the start time, the current
 * time as the end time and the current thread ID. The ProfilingResult is then
 * registered with the Profiler and the thread-local depth is decreased since
 * we have just exited a scope.
 */
ProfilerTimer::~ProfilerTimer()
{
    if (!_running) return;
    _running = false;

    auto     endTimePoint = std::chrono::high_resolution_clock::now();
    uint64_t start        = std::chrono::time_point_cast<std::chrono::microseconds>(_startPoint).time_since_epoch().count();
    uint64_t end          = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();

    ProfilingResult result{_name, _depth, start, end, threadId};
    Profiler::instance().recordResult(result);
    threadDepth--;
}
//-----------------------------------------------------------------------------