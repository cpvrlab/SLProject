#ifndef STATE_H
#define STATE_H

#include <thread>
#include <atomic>

class State
{
public:
    ~State()
    {
        _startThread.detach();
    }
    //! asynchronous start
    void start()
    {
        if (_started)
            return;

        _startThread = std::thread(&State::doStart, this);
    }

    //! if ready the state machine can change to this state
    bool started() { return _started; }
    //! signalizes that state is ready and wants caller to switch to another state
    bool ready() { return _ready; }
    void setStateReady() { _ready = true; }

    //! update this state
    virtual bool update() = 0;

protected:
    //! implement startup functionality here. Set _started to true when done.
    virtual void doStart() = 0;

    //set to true if startup is done
    bool _started = false;

private:
    //! signalizes that state is ready and wants caller to switch to another state
    bool _ready = false;

    std::thread _startThread;
};

#endif //WAI_STATE_H
