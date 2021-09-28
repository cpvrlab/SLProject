#ifndef SLPROJECT_CVCAPTUREPROVIDER_H
#define SLPROJECT_CVCAPTUREPROVIDER_H

#include <cv/CVTypedefs.h>
#include <SL.h>

#include <utility>

class CVCaptureProvider
{
private:
    SLstring _name;

protected:
    CVMat  _lastFrameBGR;
    CVMat  _lastFrameGray;
    CVSize _captureSize;

public:
    CVCaptureProvider(SLstring name) : _name(std::move(name)) {}

    virtual ~CVCaptureProvider() noexcept = default;
    virtual void   open()                 = 0;
    virtual void   grab()                 = 0;
    virtual void   close()                = 0;
    virtual SLbool isOpened()             = 0;

    SLstring name() { return _name; }
    CVMat    lastFrameBGR() { return _lastFrameBGR; }
    CVMat    lastFrameGray() { return _lastFrameGray; }
    CVSize   captureSize() { return _captureSize; }
};

#endif // SLPROJECT_CVCAPTUREPROVIDER_H
