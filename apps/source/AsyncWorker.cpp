#include "AsyncWorker.h"
#include <Utils.h>

AsyncWorker::~AsyncWorker()
{
    stop();
}

void AsyncWorker::start()
{
    _ready  = false;
    _thread = std::thread(&AsyncWorker::run, this);
}

void AsyncWorker::stop()
{
    _stop = true;
    if (_thread.joinable())
        _thread.join();
    _stop  = false;
    _ready = false;
}

//! if returns true, results are valid to be retrieved
bool AsyncWorker::isReady()
{
    return _ready;
}

bool AsyncWorker::stopRequested()
{
    return _stop;
}

// call set ready when custom run finished
void AsyncWorker::setReady()
{
    _ready = true;
}

void AsyncWorker::run()
{
    int n = 120;
    int i = 0;
    // do task
    while (true)
    {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
        Utils::log("AsyncWorker", "run til %d: %i", n, i);
        if (stopRequested())
        {
            Utils::log("AsyncWorker", "stop requested");
            break;
        }
        if (i == n)
            break;

        i++;
    }

    // task is ready
    Utils::log("AsyncWorker", "run ready");
    setReady();
}
