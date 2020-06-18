#include <SENSCamera.h>
#import "SENSiOSCameraDelegate.h"

class SENSiOSCamera : public SENSCameraBase
{
public:
    SENSiOSCamera();
    ~SENSiOSCamera();
    
    void                                   start(SENSCameraConfig config) override;
    void                                   start(std::string id, int width, int height) override;
    void                                   stop() override;
    std::vector<SENSCameraCharacteristics> getAllCameraCharacteristics() override;
    SENSFramePtr                           getLatestFrame() override;

private:
    SENSiOSCameraDelegate* _cameraDelegate;
};
