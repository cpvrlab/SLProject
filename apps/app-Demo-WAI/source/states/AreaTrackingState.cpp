#include <states/AreaTrackingState.h>

bool AreaTrackingState::update()
{
    return false;
}

void AreaTrackingState::doStart()
{
    _started = true;
}
