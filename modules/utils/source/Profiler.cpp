//
// Created by vwm1 on 09/12/2021.
//

#include <Profiler.h>
#include <iostream>
#include <cstdlib>

thread_local int Profiler::_depth = 0;

void Profiler::beginSession(std::string filePath)
{
    _filePath = std::move(filePath);
}

void Profiler::endSession()
{
    std::ofstream fileStream(_filePath, std::ios::binary);

    // -------------------------------------
    // Collect function names and thread IDs
    // -------------------------------------

    std::vector<const char*> functionNames;
    std::vector<int>         threadIds;

    for (ProfilingResult& result : _results)
    {
        if (std::find(functionNames.begin(), functionNames.end(), result.name) == functionNames.end())
            functionNames.push_back(result.name);

        if (std::find(threadIds.begin(), threadIds.end(), result.threadId) == threadIds.end())
            threadIds.push_back(result.threadId);
    }

    // ----------------------
    // Write function section
    // ----------------------

    // Write the number of functions
    writeInt32(fileStream, (uint32_t)functionNames.size());

    // Write each function name
    for (const char* functionName : functionNames)
    {
        writeInt32(fileStream, (uint32_t)std::strlen(functionName));
        fileStream << functionName;
    }

    // -------------------
    // Write trace section
    // -------------------

    // Write number of threads
    writeInt32(fileStream, (uint32_t)threadIds.size());

    for (int threadId : threadIds)
    {
        // Write thread ID
        writeInt32(fileStream, (uint32_t)threadId);
        std::cout << threadId << std::endl;

        // Count and write number of functions of thread
        int numFunctions = 0;
        for (ProfilingResult& result : _results)
        {
            if (result.threadId == threadId) numFunctions++;
        }
        writeInt32(fileStream, (uint32_t)numFunctions);

        // Write results of thread
        for (ProfilingResult& result : _results)
        {
            if (result.threadId != threadId) continue;

            auto nameIndex = std::find(functionNames.begin(), functionNames.end(), result.name) - functionNames.begin();
            auto depth     = result.depth;
            auto start     = result.start;
            auto end       = result.end;

            writeInt32(fileStream, (uint32_t)nameIndex);
            writeInt32(fileStream, depth);
            writeInt64(fileStream, start);
            writeInt64(fileStream, end);
        }
    }

    std::system((R"(start javaw -jar C:\Users\vwm1\Desktop\Traced\traceviewer-1.0.jar )" + _filePath).c_str());
}

void Profiler::recordResult(ProfilingResult result)
{
    _mutex.lock();
    _results.push_back(result);
    _mutex.unlock();
}

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

const int ONE = 1;

bool Profiler::isLittleEndian()
{
    return *(char*)(&ONE) == 1;
}

ProfilerTimer::ProfilerTimer(const char* name)
{
    _name       = name;
    _startPoint = std::chrono::high_resolution_clock::now();
    _depth      = Profiler::instance().depth();
    Profiler::instance().incDepth();
    _running = true;
}

ProfilerTimer::~ProfilerTimer()
{
    if (!_running) return;

    _running = false;

    auto      endTimePoint = std::chrono::high_resolution_clock::now();
    long long start        = std::chrono::time_point_cast<std::chrono::microseconds>(_startPoint).time_since_epoch().count();
    long long end          = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();
    int       threadId     = (int)std::hash<std::thread::id>{}(std::this_thread::get_id());

    ProfilingResult result{_name, _depth, (uint64_t)start, (uint64_t)end, threadId};
    Profiler::instance().recordResult(result);
    Profiler::instance().decDepth();
}