#ifndef SENS_ANDROID_PERMISSIONS
#define SENS_ANDROID_PERMISSIONS
#include <SENSPermissions.h>
#include <jni.h>

class SENSAndroidPermissions : public SENSPermissions
{
public:
    SENSAndroidPermissions(JavaVM* jvm);
    void askPermissions();

    bool hasCameraPermission();
    bool hasGPSPermission();
    bool hasInternetPermission();
    bool hasStoragePermission();

    bool canShowCameraPermissionDialog();
    bool canShowGPSPermissionDialog();
    bool canShowInternetPermissionDialog();
    bool canShowStoragePermissionDialog();

    bool isLocationEnabled();
    void askEnabledLocation();

private:
    JavaVM* _jvm;
};

#endif
