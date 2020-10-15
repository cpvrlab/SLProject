#ifndef SENS_IOSGPS_H
#define SENS_IOSGPS_H

#include <sens/SENSGps.h>
#import "SENSiOSGpsDelegate.h"

class SENSiOSGps : public SENSGps
{
public:
    SENSiOSGps();

    bool start() override;
    void stop() override;

private:
    //callback from delegate
    void updateLocation(double latitudeDEG,
                        double longitudeDEG,
                        double altitudeM,
                        double accuracyM);
    //callback for permission update
    void updatePermission(bool granted);
    
    SENSiOSGpsDelegate* _gpsDelegate;
};

#endif
