#include <states/StartUpState.h>

bool StartUpState::update()
{
    return false;
}

void StartUpState::doStart()
{
    _started = true;
}
