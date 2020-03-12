#ifndef AREA_TRACKING_STATE_H
#define AREA_TRACKING_STATE_H

#include <states/State.h>

class AreaTrackingState : public State
{
public:
    bool update() override;

protected:
    void doStart() override;
};

#endif //AREA_TRACKING_STATE_H
