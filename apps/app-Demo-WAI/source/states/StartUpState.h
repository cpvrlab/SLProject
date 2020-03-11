#ifndef STARTUP_STATE_H
#define STARTUP_STATE_H

#include <states/State.h>

class StartUpState : public State
{
public:
    bool update() override;

protected:
    void doStart() override;
};

#endif //STARTUP_STATE_H
