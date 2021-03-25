#include "SENSiOSPermissions.h"

SENSiOSPermissions::SENSiOSPermissions() {
    _gpsDelegate = [[SENSiOSGpsDelegate alloc] init];
}

void SENSiOSPermissions::askPermissions() {
    // TODO: implement
}

bool SENSiOSPermissions::hasCameraPermission() {
    // TODO: implement
    return true;
}

bool SENSiOSPermissions::hasGPSPermission() {
    bool result = [_gpsDelegate hasPermission];
    return result;
}

bool SENSiOSPermissions::hasInternetPermission() {
    // TODO: implement
    return true;
}

bool SENSiOSPermissions::hasStoragePermission() {
    // TODO: implement
    return true;
}

bool SENSiOSPermissions::canShowCameraPermissionsDialog() {
    // TODO: implement
    return true;
}

bool SENSiOSPermissions::canShowGPSPermissionsDialog() {
    // TODO: implement
    return true;
}

bool SENSiOSPermissions::canShowInternetPermissionsDialog() {
    // TODO: implement
    return true;
}

bool SENSiOSPermissions::canShowStoragePermissionsDialog() {
    // TODO: implement
    return true;
}
