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

#pragma mark - ARSessionDelegate

- (void)session:(ARSession *)session didUpdateFrame:(ARFrame *)frame
{
    //copy the image as in camera
    CVImageBufferRef pixelBuffer = frame.capturedImage;

    CVPixelBufferLockBaseAddress(pixelBuffer, 0);

    int            imgWidth  = (int)CVPixelBufferGetWidth(pixelBuffer);
    int            imgHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
    unsigned char* data      = (unsigned char*)CVPixelBufferGetBaseAddress(pixelBuffer);

    if (!data)
    {
        NSLog(@"No pixel buffer data");
        return;
    }
    
    //copy camera pose
    simd_float4x4 camPose = frame.camera.transform;
    //copy intrinsic
    simd_float3x3 intrinsic = frame.camera.intrinsics;
    
    //send everything via callback

    if (_updateCB)
    {
         _updateCB(&camPose,
                   data,
                   imgWidth, //frame.camera.imageResolution.width,
                   imgHeight,//frame.camera.imageResolution.height,
                   &intrinsic);
    }
}

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
