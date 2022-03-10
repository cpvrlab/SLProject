//#############################################################################
//  File:      AppPenTracking.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_APPPENTRACKING_H
#define SLPROJECT_APPPENTRACKING_H

#include <CVCaptureProvider.h>
#include <CVCaptureProviderStandard.h>
#include <CVCaptureProviderIDSPeak.h>

#include <SLVec3.h>
#include <TrackedPen.h>
#include <SLSceneView.h>

#include <app/AppPenTrackingCalibrator.h>

#include <vector>
#include <map>

//-----------------------------------------------------------------------------
typedef std::vector<CVCaptureProvider*>          CVVCaptureProvider;
typedef std::map<CVCaptureProvider*, CVTracked*> ArucoPenTrackers;
//-----------------------------------------------------------------------------
class AppPenTracking
{
public:
    static AppPenTracking& instance()
    {
        static AppPenTracking instance;
        return instance;
    }

public:
    void openCaptureProviders();
    void initTrackingSystem();
    void closeCaptureProviders();
    void grabFrameImagesAndTrack(SLSceneView* sv);
    void publishTipPose();

    // Getters
    CVVCaptureProvider&       captureProviders() { return _captureProviders; }
    CVCaptureProvider*        currentCaptureProvider() const { return _currentCaptureProvider; }
    ArucoPenTrackers&         trackers() { return _trackers; }
    bool                      doMultiTracking() const { return _doMultiTracking; }
    TrackedPen&               trackedPen() { return _trackedPen; }
    SLNode*                   penNode() { return _trackedPen.trackingSystem() ? _trackedPen.trackingSystem()->penNode() : nullptr; }
    AppPenTrackingCalibrator& calibrator() { return _calibrator; }

    // Setters
    void currentCaptureProvider(CVCaptureProvider* captureProvider) { _currentCaptureProvider = captureProvider; }
    void doMultiTracking(bool doMultiTracking) { _doMultiTracking = doMultiTracking; }

    SLVVec3f     tipPositions;
    SLGLTexture* videoTexture = nullptr;

private:
    void openCaptureProvider(CVCaptureProvider* captureProvider);
    void grabFrameImageAndTrack(CVCaptureProvider* provider, SLSceneView* sv);
    void grabFrameImage(CVCaptureProvider* provider, SLSceneView* sv);

    CVVCaptureProvider _captureProviders;
    CVCaptureProvider* _currentCaptureProvider = nullptr;
    ArucoPenTrackers   _trackers;
    bool               _doMultiTracking = true;

    TrackedPen _trackedPen;

    AppPenTrackingCalibrator _calibrator;
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_APPPENTRACKING_H
