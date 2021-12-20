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
#include <thread>

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

    // -------------------------------------
    // Collect scope names and thread IDs
    // -------------------------------------

    std::vector<const char*> scopeNames;
    std::vector<uint32_t>    threadIds;

    for (ProfilingResult& result : _results)
    {
        if (std::find(scopeNames.begin(), scopeNames.end(), result.name) == scopeNames.end())
            scopeNames.push_back(result.name);

        if (std::find(threadIds.begin(), threadIds.end(), result.threadId) == threadIds.end())
            threadIds.push_back(result.threadId);
    }

    // ----------------------
    // Write scope section
    // ----------------------

    // Write the number of scopes
    writeInt32(fileStream, (uint32_t)scopeNames.size());

    // Write each scope name
    for (const char* scopeName : scopeNames)
    {
        writeString(fileStream, scopeName);
    }

    // -------------------
    // Write trace section
    // -------------------

    // Write number of threads
    writeInt32(fileStream, (uint32_t)threadIds.size());

    for (uint32_t threadId : threadIds)
    {
        // Write thread name
        // If there was no thread name specified, generate a name of the form Thread#ID
        if (_threadNames.find(threadId) != _threadNames.end())
            writeString(fileStream, _threadNames[threadId].c_str());
        else
            writeString(fileStream, ("Thread#" + std::to_string(threadId)).c_str());

        // Count and write number of scopes in thread
        uint32_t numScopes = 0;
        for (ProfilingResult& result : _results)
        {
            if (result.threadId == threadId) numScopes++;
        }
        writeInt32(fileStream, numScopes);

        // Write results of thread
        for (ProfilingResult& result : _results)
        {
            if (result.threadId != threadId) continue;

            auto nameIndex = std::find(scopeNames.begin(), scopeNames.end(), result.name) - scopeNames.begin();
            auto depth     = result.depth;
            auto start     = result.start - _sessionStart;
            auto end       = result.end - _sessionStart;

            writeInt32(fileStream, (uint32_t)nameIndex);
            writeInt32(fileStream, depth);
            writeInt64(fileStream, start);
            writeInt64(fileStream, end);
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
 * If this function was never called for a thread at session end, the name is
 * "Thread#ID" (where ID is the signed int32 hash of the thread ID).
 * This function is sensibly also thread-safe.
 * @param name
 */
void Profiler::nameCurrentThread(const std::string& name)
{
    _mutex.lock();
    uint32_t threadId = (uint32_t)std::hash<std::thread::id>{}(std::this_thread::get_id());
    _threadNames.insert({threadId, name});
    _mutex.unlock();
}
//-----------------------------------------------------------------------------
//! Converts a 32-bit integer to big-endian and writes it to the file stream
void Profiler::writeInt32(std::ofstream& stream, uint32_t i)
{
    uint32_t bigEndian;
    if (isLittleEndian())
        bigEndian = ((i & 0x000000FF) << 24) |
                    ((i & 0x0000FF00) << 8) |
                    ((i & 0x00FF0000) >> 8) |
                    ((i & 0xFF000000) >> 24);
    else
        bigEndian = i;

    stream.write((char*)&bigEndian, 4);
}
//-----------------------------------------------------------------------------
//! Converts a 64-bit integer to big-endian and writes it to the file stream
void Profiler::writeInt64(std::ofstream& stream, uint64_t i)
{
    uint64_t bigEndian;
    if (isLittleEndian())
        bigEndian = ((i & 0x00000000000000FF) << 56) |
                    ((i & 0x000000000000FF00) << 40) |
                    ((i & 0x0000000000FF0000) << 24) |
                    ((i & 0x00000000FF000000) << 8) |
                    ((i & 0x000000FF00000000) >> 8) |
                    ((i & 0x0000FF0000000000) >> 24) |
                    ((i & 0x00FF000000000000) >> 40) |
                    ((i & 0xFF00000000000000) >> 56);
    else
        bigEndian = i;

    stream.write((char*)&bigEndian, 8);
}
//-----------------------------------------------------------------------------
//! Writes the length (32-bit) and the string (non-null-terminated) itself to the file stream
void Profiler::writeString(std::ofstream& stream, const char* s)
{
    writeInt32(stream, (uint32_t)std::strlen(s));
    stream << s;
}
//-----------------------------------------------------------------------------
/*!
 * Determines whether this machine uses the little-endian format.
 * The algorithm exploits the difference between the storage layout of the
 * 32-bit integer 1 in big-endian and little-endian.
 * Big-endian: 0x00 0x00 0x00 0x01
 * Little-endian: 0x01 0x00 0x00 0x00
 * The address of the first block is taken, converted to a uint8_t pointer and
 * dereferenced. On a little-endian machine this yields 1, on a big-endian machine 0.
 * Source: https://stackoverflow.com/questions/1001307/detecting-endianness-programmatically-in-a-c-program
 */
const uint32_t ONE = 1;
bool           Profiler::isLittleEndian()
{
    return *(uint8_t*)(&ONE) == 1;
}
//-----------------------------------------------------------------------------
thread_local int ProfilerTimer::threadDepth = 0;
//-----------------------------------------------------------------------------
/*!
 * Constructor for ProfilerTimer that saves the current time as the start
 * time, the thread-local depth as the scope depth and increases the
 * thread-local depth since we have just entered a scope.
 * @param name Name of the scope
 */
ProfilerTimer::ProfilerTimer(const char* name)
{
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
    uint32_t threadId     = (uint32_t)std::hash<std::thread::id>{}(std::this_thread::get_id());

    ProfilingResult result{_name, _depth, start, end, threadId};
    Profiler::instance().recordResult(result);
    threadDepth--;
}
//-----------------------------------------------------------------------------