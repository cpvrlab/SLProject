#ifndef ASYNC_WORKER_H
#define ASYNC_WORKER_H

#include <thread>
#include <atomic>

class AsyncWorker
{
public:
    virtual ~AsyncWorker();

    void start();
    void stop();
    //! if returns true, results are valid to be retrieved
    bool isReady();

protected:
    bool stopRequested();
    // call set ready when custom run finished
    void setReady();

    virtual void run();

private:
    std::thread _thread;

    std::atomic_bool _stop{false};
    std::atomic_bool _ready{false};
};

#endif
