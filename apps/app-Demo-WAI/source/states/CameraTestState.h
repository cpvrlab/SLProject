#ifndef CAMERA_TEST_STATE_H
#define CAMERA_TEST_STATE_H

#include <states/State.h>

class CameraTestState : public State
{
public:
    bool update() override;

protected:
    void doStart() override;
};

#endif //CAMERA_TEST_STATE_H
