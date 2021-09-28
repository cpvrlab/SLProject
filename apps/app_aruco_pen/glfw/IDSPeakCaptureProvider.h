#ifndef SLPROJECT_IDSPEAKCAPTUREPROVIDER_H
#define SLPROJECT_IDSPEAKCAPTUREPROVIDER_H

#include <CVCaptureProvider.h>

class IDSPeakCaptureProvider : public CVCaptureProvider
{
private:
    bool _isOpened = false;

public:
    IDSPeakCaptureProvider();
    ~IDSPeakCaptureProvider() noexcept override;

    void open() override;
    void grab() override;
    void close() override;
    SLbool isOpened() override;
};

#endif // SLPROJECT_IDSPEAKCAPTUREPROVIDER_H
