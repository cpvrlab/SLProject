//#############################################################################
//  File:      AppArucoPen.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_APPARUCOPEN_H
#define SLPROJECT_APPARUCOPEN_H

#include <CVCaptureProvider.h>
#include <CVCaptureProviderStandard.h>
#include <CVCaptureProviderIDSPeak.h>

#include <SLVec3.h>
#include <SLArucoPen.h>
#include <SLSceneView.h>

#include <app/AppArucoPenCalibrator.h>

#include <vector>
#include <map>

//-----------------------------------------------------------------------------
typedef std::vector<CVCaptureProvider*>          CVVCaptureProvider;
typedef std::map<CVCaptureProvider*, CVTracked*> ArucoPenTrackers;
//-----------------------------------------------------------------------------
class AppArucoPen
{
public:
    static AppArucoPen& instance()
    {
        static AppArucoPen instance;
        return instance;
    }

    SLVVec3f tipPositions;

    SLGLTexture* videoTexture = nullptr;
    SLNode*      trackedNode  = nullptr;

private:
    CVVCaptureProvider _captureProviders;
    CVCaptureProvider* _currentCaptureProvider = nullptr;
    ArucoPenTrackers   _trackers;
    bool               _doMultiTracking = true;

    SLArucoPen _arucoPen;

    AppArucoPenCalibrator _calibrator;

public:
    void       openCaptureProviders();
    void       closeCaptureProviders();
    void       grabFrameImagesAndTrack(SLSceneView* sv);
    CVTracked* currentTracker();
    void       publishTipPosition();

    // Getters
    CVVCaptureProvider&    captureProviders() { return _captureProviders; }
    CVCaptureProvider*     currentCaptureProvider() const { return _currentCaptureProvider; }
    ArucoPenTrackers&      trackers() { return _trackers; }
    bool                   doMultiTracking() { return _doMultiTracking; }
    SLArucoPen&            arucoPen() { return _arucoPen; }
    AppArucoPenCalibrator& calibrator() { return _calibrator; }

    // Setters
    void currentCaptureProvider(CVCaptureProvider* captureProvider) { _currentCaptureProvider = captureProvider; }
    void doMultiTracking(bool doMultiTracking) { _doMultiTracking = doMultiTracking; }

private:
    void openCaptureProvider(CVCaptureProvider* captureProvider);
    void grabFrameImageAndTrack(CVCaptureProvider* provider, SLSceneView* sv);
    void grabFrameImage(CVCaptureProvider* provider, SLSceneView* sv);
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_APPARUCOPEN_H
