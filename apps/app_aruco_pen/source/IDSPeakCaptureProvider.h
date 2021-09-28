#ifndef SLPROJECT_IDSPEAKCAPTUREPROVIDER_H
#define SLPROJECT_IDSPEAKCAPTUREPROVIDER_H

#include <apps/app_aruco_pen/source/CVCaptureProvider.h>

class IDSPeakCaptureProvider : public CVCaptureProvider
{
private:
    SLint  _deviceIndex;
    SLbool _isOpened = false;

public:
    explicit IDSPeakCaptureProvider(SLint deviceIndex, CVSize captureSize);
    ~IDSPeakCaptureProvider() noexcept override;

    void   open() override;
    void   grab() override;
    void   close() override;
    SLbool isOpened() override;

    SLint deviceIndex() const { return _deviceIndex; }
};

#endif // SLPROJECT_IDSPEAKCAPTUREPROVIDER_H
