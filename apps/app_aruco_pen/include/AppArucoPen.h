#ifndef SLPROJECT_APPARUCOPEN_H
#define SLPROJECT_APPARUCOPEN_H

#include <CVCaptureProvider.h>
#include <CVStandardCaptureProvider.h>
#include <IDSPeakCaptureProvider.h>
#include <vector>

typedef std::vector<CVCaptureProvider*> CVVCaptureProvider;

class AppArucoPen
{
public:
    static AppArucoPen& instance()
    {
        static AppArucoPen instance;
        return instance;
    }

public:
    CVVCaptureProvider captureProviders;
    CVCaptureProvider* currentCaptureProvider;

public:
    void openCaptureProviders();
};

#endif // SLPROJECT_APPARUCOPEN_H
