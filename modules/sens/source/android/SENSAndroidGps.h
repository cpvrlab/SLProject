#ifndef SENS_ANDROID_GPS_H
#define SENS_ANDROID_GPS_H

#include <SENSGps.h>
#include <jni.h>

class SENSAndroidGps : public SENSGps
{
public:
    SENSAndroidGps(){};

    bool start() override;
    void stop() override;

    void updateLocation(double latitudeDEG,
                        double longitudeDEG,
                        double altitudeM,
                        float  accuracyM);
};

#endif
