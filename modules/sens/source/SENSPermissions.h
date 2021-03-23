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
    virtual bool canShowCameraPermissionsDialog() {return true; };
    virtual bool canShowGPSPermissionsDialog() {return true; };
    virtual bool canShowInternetPermissionsDialog() {return true; };
    virtual bool canShowStoragePermissionsDialog() {return true; };
};

#endif
