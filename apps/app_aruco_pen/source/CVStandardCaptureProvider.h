#ifndef SLPROJECT_CVSTANDARDCAPTUREPROVIDER_H
#define SLPROJECT_CVSTANDARDCAPTUREPROVIDER_H

#include <apps/app_aruco_pen/source/CVCaptureProvider.h>
#include <cv/CVTypedefs.h>
#include <cv/CVTypes.h>

class CVStandardCaptureProvider : public CVCaptureProvider
{
private:
    SLint          _deviceIndex;
    CVVideoCapture _captureDevice;
    SLbool _isOpened = false;

public:
    explicit CVStandardCaptureProvider(SLint deviceIndex, CVSize captureSize);
    ~CVStandardCaptureProvider() noexcept override;

    void   open() override;
    void   grab() override;
    void   close() override;
    SLbool isOpened() override;

    SLint deviceIndex() const { return _deviceIndex; }
};

#endif // SLPROJECT_CVSTANDARDCAPTUREPROVIDER_H
