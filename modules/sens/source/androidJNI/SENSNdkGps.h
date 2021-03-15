#ifndef SENS_NDK_GPS_H
#define SENS_NDK_GPS_H

#include <SENSGps.h>
#include <jni.h>

class SENSNdkGps : public SENSGps
{
public:
    SENSNdkGps() {};
    void init(bool granted);

    bool start() override;
    void stop() override;

    void updateLocation(double latitudeDEG,
                        double longitudeDEG,
                        double altitudeM,
                        float  accuracyM);
};

#endif
