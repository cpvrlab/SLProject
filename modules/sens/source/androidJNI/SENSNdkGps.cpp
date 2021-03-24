#include "SENSNdkGps.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>

static SENSNdkGps* gGpsPtr = nullptr;
SENSNdkGps*        GetGpsPtr()
{
    if (gGpsPtr == nullptr)
        Utils::log("SENSNdkGps", "Global gps pointer has not been initialized");
    return gGpsPtr;
}

bool SENSNdkGps::start()
{
    if (!_permissionGranted)
        return false;
    _running = true;
    return true;
}

void SENSNdkGps::stop()
{
    if (!_running)
        return;
    _running = false;
}

void SENSNdkGps::updateLocation(double latitudeDEG,
                                double longitudeDEG,
                                double altitudeM,
                                float  accuracyM)
{
    Utils::log("SENSGps", "updateLocation");
    setLocation({latitudeDEG, longitudeDEG, altitudeM, accuracyM});
}

extern "C" JNIEXPORT void JNICALL
Java_ch_cpvr_wai_SENSGps_onLocationLLA(JNIEnv* env,
                                       jclass  obj,
                                       jdouble latitudeDEG,
                                       jdouble longitudeDEG,
                                       jdouble altitudeM,
                                       jfloat  accuracyM)
{
    Utils::log("SENSGps", "onLocationLLA");
    GetGpsPtr()->updateLocation(latitudeDEG, longitudeDEG, altitudeM, accuracyM);
}
