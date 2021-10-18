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
#include <CVStandardCaptureProvider.h>
#include <IDSPeakCaptureProvider.h>

#include <SLVec3.h>
#include <SLArucoPen.h>
#include <SLSceneView.h>

#include <app/AppArucoPenCalibrator.h>

#include <vector>

//-----------------------------------------------------------------------------
typedef std::vector<CVCaptureProvider*> CVVCaptureProvider;
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
    CVTracked* tracker = nullptr;
    SLNode* trackedNode = nullptr;

private:
    CVVCaptureProvider _captureProviders;
    CVCaptureProvider* _currentCaptureProvider = nullptr;
    SLArucoPen*        _arucoPen               = nullptr;

    AppArucoPenCalibrator _calibrator;

public:
    void openCaptureProviders();
    void closeCaptureProviders();
    void grabFrame(SLSceneView* sv);
    void publishTipPosition();

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
//-----------------------------------------------------------------------------
#endif // SLPROJECT_APPARUCOPEN_H
