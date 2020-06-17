#include <SENSCamera.h>
#import "SENSiOSCameraDelegate.h"

class SENSiOSCamera : public SENSCameraBase
{
public:
    SENSiOSCamera();
    ~SENSiOSCamera();
    
    void                                          start(SENSCameraConfig config) override;
    void                                          start(std::string id, int width, int height) override;
    void                                          stop() override;
    const std::vector<SENSCameraCharacteristics>& getAllCameraCharacteristics() override;
    SENSFramePtr                                  getLatestFrame() override;

private:
    void processNewFrame(unsigned char* data, int imgWidth, int imgHeight);
    
    SENSiOSCameraDelegate* _cameraDelegate;
    
    std::mutex _processedFrameMutex;
    SENSFramePtr _processedFrame;
};
