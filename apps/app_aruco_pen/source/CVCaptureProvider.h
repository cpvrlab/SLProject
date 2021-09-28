#ifndef SLPROJECT_CVCAPTUREPROVIDER_H
#define SLPROJECT_CVCAPTUREPROVIDER_H

#include <cv/CVTypedefs.h>
#include <cv/CVCamera.h>
#include <SL.h>

#include <utility>

class CVCaptureProvider
{
private:
    SLstring _uid;
    SLstring _name;
    CVCamera _camera;
    CVSize   _captureSize;

protected:
    CVMat _lastFrameBGR;
    CVMat _lastFrameGray;

public:
    CVCaptureProvider(SLstring uid,
                      SLstring name,
                      CVSize   captureSize);

    // Methods that must be implemented by all capture providers
    virtual ~CVCaptureProvider() noexcept = default;
    virtual void   open()                 = 0;
    virtual void   grab()                 = 0;
    virtual void   close()                = 0;
    virtual SLbool isOpened()             = 0;

    // Getters
    SLstring  uid() { return _uid; }
    SLstring  name() { return _name; }
    CVCamera& camera() { return _camera; }
    CVMat     lastFrameBGR() { return _lastFrameBGR; }
    CVMat     lastFrameGray() { return _lastFrameGray; }
    CVSize    captureSize() { return _captureSize; }
};

#endif // SLPROJECT_CVCAPTUREPROVIDER_H
