#ifndef TEST_STATE_H
#define TEST_STATE_H

#include <states/State.h>

class TestState : public State
{
public:
    bool update() override;

protected:
    void doStart() override;
};

#endif //TEST_STATE_H
