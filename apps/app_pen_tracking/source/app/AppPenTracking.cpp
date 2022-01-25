//#############################################################################
//  File:      AppPenTracking.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <app/AppPenTracking.h>

#include <SL.h>
#include <GlobalTimer.h>
#include <CVCapture.h>
#include <Utils.h>
#include <Instrumentor.h>
#include <AppDemo.h>
#include <app/AppPenTrackingROSNode.h>
#include <CVCaptureProviderSpryTrack.h>
#include <TrackingSystemSpryTrack.h>

extern void trackVideo(CVCaptureProvider* provider);

//-----------------------------------------------------------------------------
void AppPenTracking::openCaptureProviders()
{
    SL_LOG("Loading capture providers...");

    // Logitech + IDS Peak
    // openCaptureProvider(new CVCaptureProviderStandard(0, CVSize(640, 480)));
    // openCaptureProvider(new CVCaptureProviderIDSPeak(0, CVSize(1920, 1280)));

    // Logitech
    // 1280x960 actually provides better results than 1920x1080
    // Are you kidding
    // openCaptureProvider(new CVCaptureProviderStandard(0, CVSize(1280, 960)));

    // Logitech + Intel
    //    openCaptureProvider(new CVCaptureProviderStandard(2, CVSize(1280, 960)));
    //    openCaptureProvider(new CVCaptureProviderStandard(1, CVSize(1280, 960)));

    // Intel + IDS Peak
    //    openCaptureProvider(new CVCaptureProviderStandard(1, CVSize(1280, 960)));

    // Every single IDS camera that is connected
    // for (int i = 0; i < IDSPeakInterface::instance().numAvailableDevices(); i++)
    // {
    //     openCaptureProvider(new CVCaptureProviderIDSPeak(i, CVSize(2768, 1840)));
    // }

    // IDS camera + Intel + SpryTrack
//    openCaptureProvider(new CVCaptureProviderIDSPeak(0, CVSize(2768, 1840)));
//    openCaptureProvider(new CVCaptureProviderStandard(0, CVSize(1280, 720)));
    openCaptureProvider(new CVCaptureProviderSpryTrack(CVSize(1280, 720)));

    if (_captureProviders.empty())
    {
        SL_EXIT_MSG("No capture provider was opened");
    }

    _currentCaptureProvider = _captureProviders[0];

    SL_LOG("Opening done");
}
//-----------------------------------------------------------------------------
void AppPenTracking::openCaptureProvider(CVCaptureProvider* captureProvider)
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
void AppPenTracking::closeCaptureProviders()
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
void AppPenTracking::grabFrameImagesAndTrack(SLSceneView* sv)
{
    CVCapture::instance()->camSizes.clear();

    // Grab and track all non-displayed capture providers if we are in the ArUco cube scene
    if (AppPenTracking::doMultiTracking() && (AppDemo::sceneID == SID_VideoTrackArucoCubeMain || AppDemo::sceneID == SID_VirtualTrackedPen))
    {
        for (CVCaptureProvider* provider : _captureProviders)
        {
            if (_arucoPen.trackingSystem()->isAcceptedProvider(provider) && provider != _currentCaptureProvider)
            {
                grabFrameImageAndTrack(provider, sv);
            }
        }
    }

    // Grab and track the currently displayed capture provider
    if (_arucoPen.trackingSystem()->isAcceptedProvider(_currentCaptureProvider))
    {
        grabFrameImageAndTrack(_currentCaptureProvider, sv);
    }

    if (AppDemo::sceneID == SID_VideoTrackArucoCubeMain || AppDemo::sceneID == SID_VirtualTrackedPen)
    {
        AppPenTracking::instance().arucoPen().trackingSystem()->finalizeTracking();
        if (AppPenTracking::instance().arucoPen().state() == TrackedPen::Tracing)
        {
            AppPenTracking::instance().publishTipPose();
        }
    }
}
//-----------------------------------------------------------------------------
void AppPenTracking::grabFrameImageAndTrack(CVCaptureProvider* provider, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    grabFrameImage(provider, sv);
    trackVideo(provider);
}
//-----------------------------------------------------------------------------
void AppPenTracking::grabFrameImage(CVCaptureProvider* provider, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    CVCapture::instance()->startCaptureTimeMS = GlobalTimer::timeMS();

    provider->grab();
    provider->cropToAspectRatio(sv->viewportWdivH());

    CVCapture::instance()->camSizes.push_back(provider->captureSize());

    CVCapture::instance()->lastFrame     = provider->lastFrameBGR();
    CVCapture::instance()->lastFrameGray = provider->lastFrameGray();
    CVCapture::instance()->captureSize   = provider->captureSize();
    CVCapture::instance()->format        = PF_bgr;
}
//-----------------------------------------------------------------------------
void AppPenTracking::publishTipPose()
{
    AppPenTrackingROSNode::instance().publishPose(_arucoPen.rosPosition(),
                                               _arucoPen.rosOrientation());
}
//-----------------------------------------------------------------------------