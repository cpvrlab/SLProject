#ifndef SENS_IOSORIENTATION_DELEGATE_H
#define SENS_IOSORIENTATION_DELEGATE_H

@interface SENSiOSOrientationDelegate : NSObject

- (BOOL)start;
- (void)stop;

@property (nonatomic, assign) std::function<void(float, float, float, float)> updateCB;

@end

#endif
