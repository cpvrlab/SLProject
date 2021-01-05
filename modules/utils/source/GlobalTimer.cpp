#include "GlobalTimer.h"

HighResTimer GlobalTimer::_timer;

void GlobalTimer::timerStart()
{
    _timer.start();
}

float GlobalTimer::timeS()
{
    return _timer.elapsedTimeInSec();
}

float GlobalTimer::timeMS()
{
    return _timer.elapsedTimeInMilliSec();
}
