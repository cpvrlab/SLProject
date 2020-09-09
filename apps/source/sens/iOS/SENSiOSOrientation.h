#ifndef SENS_IOSORIENTATION_H
#define SENS_IOSORIENTATION_H

#include <sens/SENSOrientation.h>
#import "SENSiOSOrientationDelegate.h"

class SENSiOSOrientation : public SENSOrientation
{
public:
    SENSiOSOrientation();

    bool start() override;
    void stop() override;

private:
    //callback from delegate
    void updateOrientation(float quatX,
                           float quatY,
                           float quatZ,
                           float quatW);
    //callback for permission update
    void updatePermission(bool granted);

    SENSiOSOrientationDelegate* _orientationDelegate;
};

#endif
