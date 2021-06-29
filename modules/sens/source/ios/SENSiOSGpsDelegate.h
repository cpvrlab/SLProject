#ifndef SENS_IOSGPS_DELEGATE_H
#define SENS_IOSGPS_DELEGATE_H

#import <CoreLocation/CoreLocation.h>

@interface SENSiOSGpsDelegate : NSObject<CLLocationManagerDelegate>

- (BOOL)start;
- (void)stop;
- (void)askPermission;

@property (nonatomic, assign) std::function<void(double, double, double, double)> updateCB;
@property (nonatomic, assign) std::function<void(bool)>                           permissionCB;
@end

#endif
