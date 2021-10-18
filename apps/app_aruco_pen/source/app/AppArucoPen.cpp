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
#include <ArucoPenPublisher.h>

extern void trackVideo();

//-----------------------------------------------------------------------------
void AppArucoPen::openCaptureProviders()
{
    SL_LOG("Loading capture providers...");

//        openCaptureProvider(new CVStandardCaptureProvider(0, CVSize(640, 480)));
//        openCaptureProvider(new IDSPeakCaptureProvider(0, CVSize(1920, 1280)));

    // 1280x960 actually provides better results than 1920x1080
    // What the hell
    openCaptureProvider(new CVStandardCaptureProvider(0, CVSize(1280, 960)));

    //    openCaptureProvider(new IDSPeakCaptureProvider(0, CVSize(1920, 1280)));

    _currentCaptureProvider = _captureProviders[0];

    SL_LOG("All capture providers opened");
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
void AppArucoPen::grabFrame(SLSceneView* sv)
{

    CVCapture::instance()->camSizes.clear();

    for(CVCaptureProvider* provider : AppArucoPen::instance().captureProviders())
    {
        CVCapture::instance()->startCaptureTimeMS = GlobalTimer::timeMS();

        provider->grab();

        CVCapture::instance()->camSizes.push_back(provider->captureSize());

        CVCapture::instance()->lastFrame     = provider->lastFrameBGR();
        CVCapture::instance()->lastFrameGray = provider->lastFrameGray();
        CVCapture::instance()->captureSize   = provider->captureSize();
        CVCapture::instance()->format        = PF_bgr;

        CVCapture::instance()->adjustForSLGrayAvailable(sv->viewportWdivH());

        trackVideo();
    }
}
//-----------------------------------------------------------------------------
void AppArucoPen::publishTipPosition()
{
    SLVec3f p = AppArucoPen::instance().arucoPen()->tipPosition();
    float data[3]{p.x, p.y, p.z};
    aruco_pen_publish((void*)data, 12);
}
