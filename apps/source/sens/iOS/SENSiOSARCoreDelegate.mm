#import "SENSiOSARCoreDelegate.h"

@interface SENSiOSARCoreDelegate () {

@private
    ARSession *_arSession;
    ARConfiguration *_arConfig;
}

@end

@implementation SENSiOSARCoreDelegate

- (id)init
{
    //Initialize the parent class(es) up the hierarchy and create self:
    self = [super init];

    //Initialize members (not necessary with ARC)
    _arSession = nil;
    _arConfig  = nil;
    if (self)
    {
        [self initARKit];
    }

    return self;
}

- (void)initARKit
{
    if(ARWorldTrackingConfiguration.isSupported)
    {
        // Create an ARSession
        _arSession = [ARSession new];
        _arSession.delegate = self;
        
        _arConfig = [ARWorldTrackingConfiguration new];
        
        for(int i=0; i < ARWorldTrackingConfiguration.supportedVideoFormats.count; ++i)
        {
            CGSize s = ARWorldTrackingConfiguration.supportedVideoFormats[i].imageResolution;
            NSLog(NSStringFromCGSize(s));
        }

        _arConfig.videoFormat = ARWorldTrackingConfiguration.supportedVideoFormats.lastObject;
    }
}

- (BOOL)isAvailable
{
    return ARWorldTrackingConfiguration.isSupported;
}

- (BOOL)start
{
    if(ARWorldTrackingConfiguration.isSupported)
    {
        [_arSession runWithConfiguration:_arConfig];
        return YES;
    }
    else
        return NO;
}

- (void)stop
{
    if(_arSession)
    {
        [_arSession pause];
    }
}

- (void)latestFrame: (cv::Mat*)pose withImg: (cv::Mat*)imgBGR AndIntrinsic: (cv::Mat*)intrinsic IsTracking: (BOOL*)isTracking
{
    //Reference the current ARFrame (I think the referenced "currentFrame" may change during this function call)
    ARFrame* frame = _arSession.currentFrame;
    ARCamera* camera = frame.camera;
    
    //copy camera pose
    *pose = cv::Mat_<float>(4, 4);
    for (int i = 0; i < 4; ++i)
    {
        simd_float4 col = camera.transform.columns[i];
        pose->at<float>(0, i) = (float)col[0];
        pose->at<float>(1, i) = (float)col[1];
        pose->at<float>(2, i) = (float)col[2];
        pose->at<float>(3, i) = (float)col[3];
    }
    
    //copy intrinsic
    *intrinsic        = cv::Mat_<double>(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        simd_float3 col             = camera.intrinsics.columns[i];
        intrinsic->at<double>(0, i) = (double)col[0];
        intrinsic->at<double>(1, i) = (double)col[1];
        intrinsic->at<double>(2, i) = (double)col[2];
    }
    
    if(frame.camera.trackingState == ARTrackingStateNormal)
        *isTracking = YES;
    else
        *isTracking = NO;
    
    //copy the image as in camera
    CVImageBufferRef pixelBuffer = frame.capturedImage;

    CVReturn ret =  CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    //This is NV12, so the order is U/V (NV12: YYYYUV NV21: YYYYVU)
    if (ret == kCVReturnSuccess && pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
    {
        size_t imgWidth  = CVPixelBufferGetWidth(pixelBuffer);
        size_t imgHeight = CVPixelBufferGetHeight(pixelBuffer);

        uint8_t* yPlane = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
             
        cv::Mat yuvImg((int)imgHeight + ((int)imgHeight / 2), (int)imgWidth, CV_8UC1, yPlane);
        cv::cvtColor(yuvImg, *imgBGR, cv::COLOR_YUV2BGR_NV12, 3);
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

#pragma mark - ARSessionDelegate

/*
- (void)session:(ARSession *)session didUpdateFrame:(ARFrame *)frame
{
    if(false)
    {

        //copy the image as in camera
        CVImageBufferRef buffer = frame.capturedImage;

        //Lock the base Address so it doesn't get changed!
        CVPixelBufferLockBaseAddress(buffer, 0);
        //Get the data from the first plane (Y)
        void *address =  CVPixelBufferGetBaseAddressOfPlane(buffer, 0);
        int bufferWidth = (int)CVPixelBufferGetWidthOfPlane(buffer,0);
        int bufferHeight = (int)CVPixelBufferGetHeightOfPlane(buffer, 0);
        int bytePerRow = (int)CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
        //Get the pixel format
        OSType pixelFormat = CVPixelBufferGetPixelFormatType(buffer);
            
        cv::Mat converted;
        //NOTE: CV_8UC3 means unsigned (0-255) 8 bits per pixel, with 3 channels!
        //Check to see if this is the correct pixel format
        if (pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
            //We have an ARKIT buffer
            //Get the yPlane (Luma values)
            //HighResTimer t;
            cv::Mat yPlane = cv::Mat(bufferHeight, bufferWidth, CV_8UC1, address);
                
            //Get cbcrPlane (Chroma values)
            int cbcrWidth = (int)CVPixelBufferGetWidthOfPlane(buffer,1);
            int cbcrHeight = (int)CVPixelBufferGetHeightOfPlane(buffer, 1);
            void *cbcrAddress = CVPixelBufferGetBaseAddressOfPlane(buffer, 1);
            //Since the CbCr Values are alternating we have 2 channels: Cb and Cr. Thus we need to use CV_8UC2 here.
            cv::Mat cbcrPlane = cv::Mat(cbcrHeight, cbcrWidth, CV_8UC2, cbcrAddress);
                
            //Split them apart so we can merge them with the luma values
            std::vector<cv::Mat> cbcrPlanes;
            cv::split(cbcrPlane, cbcrPlanes);
                
            cv::Mat cbPlane;
            cv::Mat crPlane;
                
            //Since we have a 4:2:0 format, cb and cr values are only present for each 2x2 luma pixels. Thus we need to enlargen them (by a factor of 2).
            cv::resize(cbcrPlanes[0], cbPlane, yPlane.size(), 0, 0, cv::INTER_NEAREST);
            cv::resize(cbcrPlanes[1], crPlane, yPlane.size(), 0, 0, cv::INTER_NEAREST);
                
            cv::Mat ycbcr;
            std::vector<cv::Mat> allPlanes = {yPlane, cbPlane, crPlane};
            cv::merge(allPlanes, ycbcr);
                
            //ycbcr now contains all three planes. We need to convert it from YCbCr to RGB so OpenCV can work with it
            cv::cvtColor(ycbcr, converted, cv::COLOR_YCrCb2RGB);
            //float elt = t.elapsedTimeInMilliSec();
            //   NSLog(@"convertion: %f", elt);
        }
        else
        {
            //Probably RGB so just use that.
            converted = cv::Mat(bufferHeight, bufferWidth, CV_8UC3, address, bytePerRow).clone();
        }

        //Since we clone the cv::Mat no need to keep the Buffer Locked while we work on it.
        CVPixelBufferUnlockBaseAddress(buffer, 0);
        
        //copy camera pose
        simd_float4x4 camPose = frame.camera.transform;
        //copy intrinsic
        simd_float3x3 intrinsic = frame.camera.intrinsics;
        
       
        if(_updateBgrCB)
        {
            _updateBgrCB(&camPose, converted, &intrinsic);
        }

    }
    else
    {
        //copy the image as in camera
        CVImageBufferRef pixelBuffer = frame.capturedImage;

        CVReturn ret =  CVPixelBufferLockBaseAddress(pixelBuffer, 0);
        if(ret != kCVReturnSuccess)
            return;
        
        //Get the pixel format
        OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);

        //This is NV12, so the order is U/V (NV12: YYYYUV NV21: YYYYVU)
        if (pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        {
            size_t imgWidth  = CVPixelBufferGetWidth(pixelBuffer);
            size_t imgHeight = CVPixelBufferGetHeight(pixelBuffer);
            //unsigned char* data = (unsigned char*)CVPixelBufferGetBaseAddress(pixelBuffer);

            //debug info for now
            //size_t left, right, top, bottom;
            //CVPixelBufferGetExtendedPixels(pixelBuffer, &left, &right, &top, &bottom);
            //size_t bytesPerRow = (int)CVPixelBufferGetBytesPerRow (pixelBuffer);
            
            uint8_t* yPlane = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
            uint8_t* uvPlane = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
                  
            //copy camera pose
            simd_float4x4 camPose = frame.camera.transform;
            //copy intrinsic
            simd_float3x3 intrinsic = frame.camera.intrinsics;
            
            BOOL tracking = NO;
            if(frame.camera.trackingState == ARTrackingStateNormal)
                tracking = YES;
            
            //send everything via callback
            if (_updateCB)
            {
                 _updateCB(&camPose,
                           yPlane,
                           uvPlane,
                           imgWidth,
                           imgHeight,
                           &intrinsic,
                           tracking);
            }
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    }
}
 */

- (void)session:(ARSession *)session didFailWithError:(NSError *)error
{
    // Present an error message to the user
    
}

- (void)session:(ARSession *)session cameraDidChangeTrackingState:(ARCamera *)camera
{
    switch (camera.trackingState) {
        case ARTrackingStateNormal:
            NSLog(@"Tracking is Normal.\n");
            break;
        case ARTrackingStateLimited:
            NSLog(@"Tracking is limited: ");
            switch(camera.trackingStateReason)
            {
                case ARTrackingStateReasonNone:
                    NSLog(@"Tracking is not limited.\n");
                    break;
                case ARTrackingStateReasonInitializing:
                    NSLog(@"Tracking is limited due to initialization in progress.\n");
                    break;
                case ARTrackingStateReasonExcessiveMotion:
                    NSLog(@"Tracking is limited due to a excessive motion of the camera.\n");
                    break;
                case ARTrackingStateReasonInsufficientFeatures:
                    NSLog(@"Tracking is limited due to a lack of features visible to the camera.\n");
                    break;
                case ARTrackingStateReasonRelocalizing:
                    NSLog(@"Tracking is limited due to a relocalization in progress.\n");
                    break;
                default:
                    break;
            }
            break;
        case ARTrackingStateNotAvailable:
            NSLog(@"Tracking is not available.\n");
            break;
        default:
            break;
    }
}

- (void)sessionWasInterrupted:(ARSession *)session
{
    // Inform the user that the session has been interrupted, for example, by presenting an overlay
}

- (void)sessionInterruptionEnded:(ARSession *)session
{
    // Reset tracking and/or remove existing anchors if consistent tracking is required
}

/*
//! Starts the location data update if the interval time > 0 else it stops
- (void)setupLocationManager
{
    if ([CLLocationManager locationServicesEnabled])
    {
        // Init location manager
        _locationManager                 = [[CLLocationManager alloc] init];
        _locationManager.delegate        = self;
        _locationManager.desiredAccuracy = kCLLocationAccuracyBest;
        //self.locationManager.distanceFilter = 1;

        // for iOS 8, specific user level permission is required,
        // "when-in-use" authorization grants access to the user's location.
        // important: be sure to include NSLocationWhenInUseUsageDescription along with its
        // explanation string in your Info.plist or startUpdatingLocation will not work
        if ([_locationManager respondsToSelector:@selector(requestWhenInUseAuthorization)])
        {
            [_locationManager requestWhenInUseAuthorization];
        }
    }
    else
    {
         //Location services are not enabled.
         //Take appropriate action: for instance, prompt the
         //user to enable the location services
        NSLog(@"Location services are not enabled");
    }
}

//! Starts the location data update
- (BOOL)start
{
    if (_locationManager == nil)
        return NO;

    if (!_running)
    {
        [_locationManager startUpdatingLocation];
        _running = true;
        printf("Starting Location Manager\n");
    }

    return YES;
}
//-----------------------------------------------------------------------------
//! Stops the location data update
- (void)stop
{
    if (_running)
    {
        [_locationManager stopUpdatingLocation];
        _running = false;
        printf("Stopping Location Manager\n");
    }
}
//-----------------------------------------------------------------------------
- (void)locationManager:(CLLocationManager*)manager didUpdateToLocation:(CLLocation*)newLocation fromLocation:(CLLocation*)oldLocation
{
    printf("horizontalAccuracy: %f\n", newLocation.horizontalAccuracy);

    // negative horizontal accuracy means no location fix
    if (newLocation.horizontalAccuracy > 0.0)
    {
        //callback here
        if (_updateCB)
        {
            _updateCB(newLocation.coordinate.latitude,
                      newLocation.coordinate.longitude,
                      newLocation.altitude,
                      newLocation.horizontalAccuracy);
        }
    }
}
//-----------------------------------------------------------------------------
- (void)locationManager:(CLLocationManager*)manager didFailWithError:(NSError*)error
{
    // The location "unknown" error simply means the manager is currently unable to get the location.
    // We can ignore this error for the scenario of getting a single location fix, because we already have a
    // timeout that will stop the location manager to save power.
    //
    if ([error code] != kCLErrorLocationUnknown)
    {
        printf("**** locationManager didFailWithError ****\n");
        [self stop];
    }
}

- (void)locationManager:(CLLocationManager*)manager didChangeAuthorizationStatus:(CLAuthorizationStatus)status
{
    //callback permission status
    if (status == kCLAuthorizationStatusAuthorizedAlways ||
        status == kCLAuthorizationStatusAuthorizedWhenInUse)
    {
        if (_permissionCB)
            _permissionCB(true);
    }
    else
    {
        if (_permissionCB)
            _permissionCB(false);
    }
}
*/
@end
