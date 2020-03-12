#ifndef TUTORIAL_STATE_H
#define TUTORIAL_STATE_H

#include <states/State.h>

class TutorialState : public State
{
public:
    bool update() override;

protected:
    void doStart() override;
};

#endif //TUTORIAL_STATE_H
