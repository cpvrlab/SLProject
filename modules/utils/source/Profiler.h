//
// Created by Magmi on 07.12.2021.
//

#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <map>
#include <thread>
#include <mutex>

#define PROFILING_NEW 0

#ifdef PROFILING_NEW
#    define NEW_PROFILE_FUNCTION() ProfilerTimer funcProfiler(__FUNCTION__)
#else
#    define NEW_PROFILE_FUNCTION()
#endif

struct ProfilingResult
{
    const char* name;
    uint32_t    depth;
    uint64_t    start;
    uint64_t    end;
    int         threadId;
};

class Profiler
{

private:
    thread_local static int _depth;

    std::string                  _filePath;
    std::vector<ProfilingResult> _results;
    std::mutex                   _mutex;

public:
    static Profiler& instance()
    {
        static Profiler instance;
        return instance;
    }

    void beginSession(std::string filePath);
    void endSession();

    int  depth() { return _depth; }
    void incDepth() { _depth++; }
    void decDepth() { _depth--; }
    void recordResult(ProfilingResult result);

private:
    static void writeInt32(std::ofstream& stream, uint32_t i);
    static void writeInt64(std::ofstream& stream, uint64_t i);
    static bool isLittleEndian();
};

class ProfilerTimer
{

private:
    const char*                                                 _name;
    uint32_t                                                    _depth;
    std::chrono::time_point<std::chrono::high_resolution_clock> _startPoint;
    bool                                                        _running;

public:
    explicit ProfilerTimer(const char* name);
    ~ProfilerTimer();
};

#endif // PROFILER_H
