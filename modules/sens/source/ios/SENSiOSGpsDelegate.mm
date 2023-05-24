#import "SENSiOSGpsDelegate.h"

@interface SENSiOSGpsDelegate () {

@private
    CLLocationManager* _locationManager;
    BOOL               _running;
}

@end

@implementation SENSiOSGpsDelegate

- (id)init
{
    //Initialize the parent class(es) up the hierarchy and create self:
    self = [super init];

    //Initialize members (not necessary with ARC)
    _locationManager = nil;
    _running         = NO;
    if (self)
    {
        [self setupLocationManager];
    }

    return self;
}

//! Starts the location data update if the interval time > 0 else it stops
- (void)setupLocationManager
{
    // Init location manager
    _locationManager                 = [[CLLocationManager alloc] init];
    _locationManager.delegate        = self;
    _locationManager.desiredAccuracy = kCLLocationAccuracyBest;

    [self askPermission];
}

- (void)askPermission
{
    // for iOS 8, specific user level permission is required,
    // "when-in-use" authorization grants access to the user's location.
    // important: be sure to include NSLocationWhenInUseUsageDescription along with its
    // explanation string in your Info.plist or startUpdatingLocation will not work
    if ([_locationManager respondsToSelector:@selector(requestWhenInUseAuthorization)])
    {
        [_locationManager requestWhenInUseAuthorization];
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
    //printf("horizontalAccuracy: %f\n", newLocation.horizontalAccuracy);

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

@end
