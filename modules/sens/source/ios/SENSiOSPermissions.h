#ifndef SENS_IOSPERMISSIONS_H
#define SENS_IOSPERMISSIONS_H

#include <SENSPermissions.h>

#import <CoreLocation/CoreLocation.h>
#import <AVFoundation/AVCaptureDevice.h>

#include "SENSiOSGps.h"

class SENSiOSPermissions : public SENSPermissions
{
public:
    SENSiOSPermissions(SENSiOSGps* gps);

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
    SENSiOSGps* _gps;
};

#endif
