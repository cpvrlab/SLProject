#ifndef SENS_NDK_PERMISSIONS
#define SENS_NDK_PERMISSIONS
#include <SENSPermissions.h>
#include <jni.h>

class SENSNdkPermissions : public SENSPermissions
{
    public:
    SENSNdkPermissions(JavaVM* jvm);
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

