#ifndef SLPROJECT_CVSTANDARDCAPTUREPROVIDER_H
#define SLPROJECT_CVSTANDARDCAPTUREPROVIDER_H

#include <CVCaptureProvider.h>
#include <cv/CVTypedefs.h>
#include <cv/CVTypes.h>
#include <cv/CVCamera.h>

class CVStandardCaptureProvider : public CVCaptureProvider
{
private:
    CVVideoCapture _captureDevice;
    CVCamera       _camera;

public:
    CVStandardCaptureProvider();
    ~CVStandardCaptureProvider() noexcept override;

    void open() override;
    void grab() override;
    void close() override;
    SLbool isOpened() override;

    CVCamera camera() { return _camera; }
};

#endif // SLPROJECT_CVSTANDARDCAPTUREPROVIDER_H
