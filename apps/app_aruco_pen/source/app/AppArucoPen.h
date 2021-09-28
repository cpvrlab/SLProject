#ifndef SLPROJECT_APPARUCOPEN_H
#define SLPROJECT_APPARUCOPEN_H

#include <CVCaptureProvider.h>
#include <CVStandardCaptureProvider.h>
#include <IDSPeakCaptureProvider.h>

#include <SLArucoPen.h>

#include <app/AppArucoPenCalibrator.h>

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

private:
    CVVCaptureProvider _captureProviders;
    CVCaptureProvider* _currentCaptureProvider = nullptr;
    SLArucoPen*        _arucoPen               = nullptr;

    AppArucoPenCalibrator _calibrator;

public:
    void openCaptureProviders();
    void closeCaptureProviders();

    // Getters
    CVVCaptureProvider&    captureProviders() { return _captureProviders; }
    CVCaptureProvider*     currentCaptureProvider() const { return _currentCaptureProvider; }
    SLArucoPen*            arucoPen() const { return _arucoPen; }
    AppArucoPenCalibrator& calibrator() { return _calibrator; }

    // Setters
    void currentCaptureProvider(CVCaptureProvider* captureProvider) { _currentCaptureProvider = captureProvider; }
    void arucoPen(SLArucoPen* arucoPen) { _arucoPen = arucoPen; }

private:
    void openCaptureProvider(CVCaptureProvider* captureProvider);
};

#endif // SLPROJECT_APPARUCOPEN_H
