#include "SENSiOSARCore.h"

SENSiOSARCore::SENSiOSARCore()
{
    _arcoreDelegate = [[SENSiOSARCoreDelegate alloc] init];
    //check availablity
    _available = [_arcoreDelegate isAvailable];
    //set update callback
    [_arcoreDelegate setUpdateCB:std::bind(&SENSiOSARCore::onUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5)];
    
    /*
    _gpsDelegate = [[SENSiOSGpsDelegate alloc] init];
    //set update callback
    [_gpsDelegate setUpdateCB:std::bind(&SENSiOSGps::updateLocation, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)];
    //set permission callback
    [_gpsDelegate setPermissionCB:std::bind(&SENSiOSGps::updatePermission, this, std::placeholders::_1)];
     */
}

bool SENSiOSARCore::init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray)
{
    if(!_available)
        return false;
    
    bool success = [_arcoreDelegate start];
    
    return success;
}

bool SENSiOSARCore::isReady()
{
    return false;
}

bool SENSiOSARCore::resume()
{
    return false;
}

void SENSiOSARCore::reset()
{
    
}

void SENSiOSARCore::pause()
{
    
}

bool SENSiOSARCore::update(cv::Mat& intrinsic, cv::Mat& view)
{
    return false;
}

SENSFramePtr SENSiOSARCore::latestFrame()
{
    return nullptr;
}

void SENSiOSARCore::setDisplaySize(int w, int h)
{
    
}

void SENSiOSARCore::onUpdate(simd_float4x4* camPose, unsigned char* data, int imgWidth, int imgHeight, simd_float3x3* camMat3x3)
{
    
}

/*
bool SENSiOSGps::start()
{
    if (!_permissionGranted)
        return false;
    else
    {
        _running = [_gpsDelegate start];
        return _running;
    }
}

void SENSiOSGps::stop()
{
    [_gpsDelegate stop];
    _running = false;
}

void SENSiOSGps::updateLocation(double latitudeDEG,
                                double longitudeDEG,
                                double altitudeM,
                                double accuracyM)
{
    setLocation({latitudeDEG, longitudeDEG, altitudeM, (float)accuracyM});
}

void SENSiOSGps::updatePermission(bool granted)
{
    _permissionGranted = granted;
}
*/
