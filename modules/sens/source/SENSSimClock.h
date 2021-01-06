#ifndef SENS_SIMCLOCK_H
#define SENS_SIMCLOCK_H

#include <mutex>
#include <atomic>
#include <SENS.h>

/*! SENSSimClock
Clock used for sensor simulation in SENSSimulator and SENSSimulated
*/
class SENSSimClock
{
public:
    SENSSimClock(SENSTimePt startTimePt, SENSTimePt simStartTimePt)
      : _startTimePt(startTimePt),
        _simStartTimePt(simStartTimePt)
    {
        _pause       = true;
        _pauseTimePt = (SENSTimePt)SENSClock::now();
    }

    //!pause simulation
    void pause()
    {
        std::lock_guard<std::mutex> lock(_pauseMutex);
        _pause       = true;
        _pauseTimePt = (SENSTimePt)SENSClock::now();
    }

    bool isPaused() const
    {
        return _pause;
    }

    //!resume simulation
    void resume()
    {
        std::lock_guard<std::mutex> lock(_pauseMutex);
        _pause     = false;
        _pauseTime = _pauseTime + std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - _pauseTimePt);
    }

    //!reset simulation
    //!(we switch to pause when setting new start, in this way now becomes thread save and we increase performance of now() (no mutex in nomal use))
    void reset()
    {
        pause();
        _startTimePt = SENSClock::now();
        _pauseTime   = SENSMicroseconds(0);
        resume();
    }

    //!get passed simulation time
    SENSMicroseconds passedTime() const
    {
        SENSMicroseconds passedSimTime;
        if (_pause)
        {
            const std::lock_guard<std::mutex> lock(_pauseMutex);
            passedSimTime = std::chrono::duration_cast<SENSMicroseconds>(_pauseTimePt - _startTimePt) - _pauseTime;
        }
        else
        {
            passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - _startTimePt) - _pauseTime;
        }

        return passedSimTime;
    }

    //!get current simulation time (const function which should be thread save)
    SENSTimePt now() const
    {
        SENSMicroseconds passedSimTime;
        if (_pause)
        {
            const std::lock_guard<std::mutex> lock(_pauseMutex);
            passedSimTime = std::chrono::duration_cast<SENSMicroseconds>(_pauseTimePt - _startTimePt) - _pauseTime;
        }
        else
        {
            passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - _startTimePt) - _pauseTime;
        }

        return _simStartTimePt + passedSimTime;
    }

private:
    //!atomic, because we use it in now without mutex
    std::atomic_bool _pause{true};
    SENSTimePt       _pauseTimePt;
    //(mutable as we want to use it in the const function now())
    mutable std::mutex _pauseMutex;

    //!Offset from pause state: we have to accumulate pause times and subtract it from sim time
    SENSMicroseconds _pauseTime{0};
    //!real start time point in the present (when SENSSimulation::start was called)
    SENSTimePt _startTimePt;
    //!start time point of simulation in the past
    SENSTimePt _simStartTimePt;
};

#endif //SENS_SIMCLOCK_H
