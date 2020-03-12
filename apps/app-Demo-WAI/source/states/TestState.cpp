#include <states/TestState.h>

bool TestState::update()
{
    return false;
}

void TestState::doStart()
{
    _started = true;
}
