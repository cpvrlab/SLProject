#ifndef SENS_PERMISSIONS
#define SENS_PERMISSIONS

class SENSPermissions
{
public:
    virtual void askPermissions() {}
    virtual bool hasCameraPermission() { return true; }
    virtual bool hasGPSPermission() { return true; }
    virtual bool hasInternetPermission() { return true; }
    virtual bool hasStoragePermission() { return true; }
    virtual bool canShowCameraPermissionDialog() { return true; };
    virtual bool canShowGPSPermissionDialog() { return true; };
    virtual bool canShowInternetPermissionDialog() { return true; };
    virtual bool canShowStoragePermissionDialog() { return true; };
    virtual bool isLocationEnabled() { return true; }
    virtual void askEnabledLocation(){};
};

#endif
