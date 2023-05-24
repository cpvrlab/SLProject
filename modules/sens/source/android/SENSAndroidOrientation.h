#ifndef SENS_ANDROID_ORIENTATION_H
#define SENS_ANDROID_ORIENTATION_H

#include <SENSOrientation.h>
#include <jni.h>

class SENSAndroidOrientation : public SENSOrientation
{
public:
    SENSAndroidOrientation(){};
    void init(bool granted){};

    bool start() override;
    void stop() override;

    void updateOrientation(const Quat& orientation);
};

#endif
