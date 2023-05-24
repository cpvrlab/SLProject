#import "SENSiOSOrientationDelegate.h"
#import <CoreMotion/CoreMotion.h>
#import <GLKit/GLKit.h>

#include <SENSUtils.h>

@interface SENSiOSOrientationDelegate () {

@private
    CMMotionManager* _motionManager;
    BOOL             _running;
}

@end

@implementation SENSiOSOrientationDelegate

- (id)init
{
    //Initialize the parent class(es) up the hierarchy and create self:
    self = [super init];
    //Initialize members (not necessary with ARC)
    _motionManager = nil;
    _running       = NO;
    if (self)
    {
        _motionManager = [[CMMotionManager alloc] init];
    }

    return self;
}

- (BOOL)start
{
    if (_motionManager == nil)
        return NO;

    if (!_running)
    {
        [self startMotionManager:1.0 / 20.0];
        //[_motionManager startDeviceMotionUpdates];
        _running = true;
        printf("Starting Motion Manager\n");
    }

    return YES;
}

- (void)stop
{
    if (_running)
    {
        [_motionManager stopDeviceMotionUpdates];
        _running = false;
        printf("Stopping Location Manager\n");
    }
}

//-----------------------------------------------------------------------------
//! Starts the motion data update if the interval time > 0 else it stops
- (void)startMotionManager:(double)intervalTimeSEC
{
    if ([_motionManager isDeviceMotionAvailable] == YES)
    {
        _motionManager.deviceMotionUpdateInterval = intervalTimeSEC;

        // See also: https://developer.apple.com/documentation/coremotion/getting_processed_device_motion_data/understanding_reference_frames_and_device_attitude?language=objc

        // In case the compass calibration is turned off on the iOS device,
        // the following code receives an error with code 102 - true north not available.
        // This is why we use the magnetic north here.
        [_motionManager startDeviceMotionUpdatesUsingReferenceFrame:CMAttitudeReferenceFrameXMagneticNorthZVertical
                                                            toQueue:[NSOperationQueue currentQueue]
                                                        withHandler:^(CMDeviceMotion* motion, NSError* error) {
                                                          [self performSelectorOnMainThread:@selector(onDeviceMotionUpdate:)
                                                                                 withObject:motion
                                                                              waitUntilDone:YES];
                                                        }];
    }
    else
    {
        [_motionManager stopDeviceMotionUpdates];
    }
}
//-----------------------------------------------------------------------------
- (void)onDeviceMotionUpdate:(CMDeviceMotion*)motion
{
    CMDeviceMotion* motionData = _motionManager.deviceMotion;
    CMAttitude*     attitude   = motionData.attitude;

    //Get sensor rotation as quaternion. This quaternion describes a rotation relative to NWU-frame
    //(see: https://developer.apple.com/documentation/coremotion/getting_processed_device_motion_data/understanding_reference_frames_and_device_attitude)
    CMQuaternion q = attitude.quaternion;

    /*
    //test w.r.t. to NWU-frame
    if(_updateCB)
        _updateCB(q.x, q.y, q.z, q.w);

    printf("Rotation: xRot %f yRot %f zRot %f\n", attitude.pitch * SENS_RAD2DEG, attitude.roll * SENS_RAD2DEG, attitude.yaw * SENS_RAD2DEG);
    */

    //(https://developer.apple.com/documentation/coremotion/getting_processed_device-motion_data/understanding_reference_frames_and_device_attitude)
    //We configure CMMotionManager with xMagneticNorthZVertical which means its a frame, where x points north, y points west and z points up (NWU).
    //We add rotation of 90 deg. around z-axis to relate the sensor rotation to an ENU-frame (as in Android)
    GLKQuaternion qNWU    = GLKQuaternionMake(q.x, q.y, q.z, q.w);
    GLKQuaternion qRot90Z = GLKQuaternionMakeWithAngleAndAxis(GLKMathDegreesToRadians(90), 0, 0, 1);
    GLKQuaternion qENU    = GLKQuaternionMultiply(qRot90Z, qNWU);

    // Send quaternion
    if (_updateCB)
        _updateCB(qENU.q[0], qENU.q[1], qENU.q[2], qENU.q[3]);
}

@end
