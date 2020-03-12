#include <states/TutorialState.h>

bool TutorialState::update()
{
    return false;
}

void TutorialState::doStart()
{
    _started = true;
}
