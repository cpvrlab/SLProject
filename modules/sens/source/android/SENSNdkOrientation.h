#ifndef SENS_NDK_ORIENTATION_H
#define SENS_NDK_ORIENTATION_H

#include <SENSOrientation.h>
#include <jni.h>

class SENSNdkOrientation : public SENSOrientation
{
public:
    SENSNdkOrientation(){};
    void init(bool granted) {};

    bool start() override;
    void stop() override;

    void updateOrientation(const Quat& orientation);
};

#endif
