#ifndef SENS_IOSPERMISSIONS_H
#define SENS_IOSPERMISSIONS_H

#include <SENSPermissions.h>
#include "SENSiOSGpsDelegate.h"

class SENSiOSPermissions : public SENSPermissions {
public:
    SENSiOSPermissions();
    
    void askPermissions();
    
    bool hasCameraPermission();
    bool hasGPSPermission();
    bool hasInternetPermission();
    bool hasStoragePermission();
    bool canShowCameraPermissionsDialog();
    bool canShowGPSPermissionsDialog();
    bool canShowInternetPermissionsDialog();
    bool canShowStoragePermissionsDialog();
    
private:
    SENSiOSGpsDelegate* _gpsDelegate;
};

#endif
