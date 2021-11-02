//#############################################################################
//  File:      AppArucoPen.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "AppArucoPen.h"

#include <SL.h>
#include <GlobalTimer.h>
#include <CVCapture.h>
#include <Utils.h>
#include <AppDemo.h>
#include <app/AppArucoPenROSNode.h>

extern void trackVideo(CVCaptureProvider* provider);

//-----------------------------------------------------------------------------
void AppArucoPen::openCaptureProviders()
{
    SL_LOG("Loading capture providers...");

    // Logitech + IDS Peak
    // openCaptureProvider(new CVStandardCaptureProvider(0, CVSize(640, 480)));
    // openCaptureProvider(new IDSPeakCaptureProvider(0, CVSize(1920, 1280)));

    // Logitech
    // 1280x960 actually provides better results than 1920x1080
    // Are you kidding
    // openCaptureProvider(new CVStandardCaptureProvider(0, CVSize(1280, 960)));

    // Logitech + Intel
    openCaptureProvider(new CVStandardCaptureProvider(2, CVSize(1280, 960)));
    openCaptureProvider(new CVStandardCaptureProvider(1, CVSize(640, 480)));

    _currentCaptureProvider = _captureProviders[0];

    SL_LOG("Opening done");
}
//-----------------------------------------------------------------------------
void AppArucoPen::openCaptureProvider(CVCaptureProvider* captureProvider)
{
    float before = GlobalTimer::timeS();
    captureProvider->open();

    if (!captureProvider->isOpened())
    {
        SL_LOG("Failed to open capture provider \"%s\"", captureProvider->name().c_str());
        return;
    }

    _captureProviders.push_back(captureProvider);
    float delta = GlobalTimer::timeS() - before;

    SL_LOG("%s capture provider opened in %f s",
           captureProvider->name().c_str(),
           delta);

    SLstring configPath    = AppDemo::calibFilePath;
    SLstring calibFilename = "camCalib_" + captureProvider->uid() + ".xml";
    SLstring calibPath     = Utils::unifySlashes(configPath) + calibFilename;

    if (Utils::fileExists(calibPath))
    {
        SL_LOG("Found calibration for capture at %s", calibPath.c_str());
        if (captureProvider->camera().calibration.load(configPath,
                                                       calibFilename,
                                                       true))
        {
            SL_LOG("Calibration successfully loaded");
        }
        else
        {
            SL_LOG("Failed to load calibration");
        }
    }
}
//-----------------------------------------------------------------------------
void AppArucoPen::closeCaptureProviders()
{
    SL_LOG("Closing capture providers...");

    for (CVCaptureProvider* captureProvider : _captureProviders)
    {
        captureProvider->close();
        delete captureProvider;
    }

    SL_LOG("All capture providers closed");
}
//-----------------------------------------------------------------------------
void AppArucoPen::grabFrameImagesAndTrack(SLSceneView* sv)
{
    CVCapture::instance()->camSizes.clear();

    // Grab and track all non-displayed capture providers if we are in the ArUco cube scene
    if (AppDemo::sceneID == SID_VideoTrackArucoCubeMain || AppDemo::sceneID == SID_VirtualArucoPen)
    {
        for (CVCaptureProvider* provider : _captureProviders)
        {
            if (provider != _currentCaptureProvider)
            {
                grabFrameImageAndTrack(provider, sv);
            }
        }
    }

    // Grab and track the currently displayed capture provider
    grabFrameImageAndTrack(_currentCaptureProvider, sv);

    if (AppDemo::sceneID == SID_VideoTrackArucoCubeMain || AppDemo::sceneID == SID_VirtualArucoPen)
    {
        AppArucoPen::instance().arucoPen().multiTracker().combine();
        AppArucoPen::instance().publishTipPosition();
    }
}
//-----------------------------------------------------------------------------
void AppArucoPen::grabFrameImageAndTrack(CVCaptureProvider* provider, SLSceneView* sv)
{
    CVCapture::instance()->startCaptureTimeMS = GlobalTimer::timeMS();

    provider->grab();
    provider->cropToAspectRatio(sv->viewportWdivH());

    CVCapture::instance()->camSizes.push_back(provider->captureSize());

    CVCapture::instance()->lastFrame     = provider->lastFrameBGR();
    CVCapture::instance()->lastFrameGray = provider->lastFrameGray();
    CVCapture::instance()->captureSize   = provider->captureSize();
    CVCapture::instance()->format        = PF_bgr;

    //    CVCapture::instance()->adjustForSLGrayAvailable(sv->viewportWdivH());

    trackVideo(provider);
}
//-----------------------------------------------------------------------------
CVTracked* AppArucoPen::currentTracker()
{
    if(_trackers.empty())
    {
        return nullptr;
    }

    CVTracked* tracked = _trackers.at(_currentCaptureProvider);
    return tracked;
}
//-----------------------------------------------------------------------------
void AppArucoPen::publishTipPosition()
{
    SLVec3f p = AppArucoPen::instance().arucoPen().tipPosition();
    AppArucoPenROSNode::instance().publish(p.x, p.y, p.z);
}
//-----------------------------------------------------------------------------