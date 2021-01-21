#include <CompassAlignment.h>

float CompassAlignment::calculateRotAngle(const SLDeviceLocation& devLoc,
                                          const SENSCalibration&  calibration,
                                          Template&               tpl,
                                          SLMat4f&                camPose,
                                          SENSFramePtr&           frame)
{
    float result = 0.0f;

    // world coordinate system has its origin at the center of the model
    // and its axes are aligned east-up-north
    SLVec2f pTtpl = calculateExpectedTemplatePixelLocation(devLoc, calibration, tpl.latDegLonDegAltM, camPose);

    std::cout << "pTtpl: " << pTtpl << std::endl;

    cv::Mat resultImage;
    cv::matchTemplate(frame->imgManip, tpl.image, resultImage, cv::TM_CCOEFF_NORMED);

    double    minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(resultImage, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::rectangle(frame->imgBGR, maxLoc, maxLoc + cv::Point(tpl.image.cols, tpl.image.rows), cv::Scalar(0, 0, 255));
    cv::rectangle(frame->imgBGR, cv::Point(pTtpl.x, pTtpl.y) + cv::Point(1, 1), cv::Point(pTtpl.x, pTtpl.y) + cv::Point(-1, -1), cv::Scalar(255, 0, 0));

    cv::Point tplCenter      = cv::Point(tpl.image.cols * 0.5f, tpl.image.rows * 0.5f);
    cv::Point tplMatchCenter = maxLoc + tplCenter;

    std::cout << "tplMatchCenter: " << tplMatchCenter << std::endl;

    int xOffsetPix = pTtpl.x - tplMatchCenter.x;
    if (xOffsetPix != 0)
    {
        float focalLength = calibration.fx();

        //calculate the offset matrix:
        result = atanf((float)xOffsetPix / focalLength);

        std::cout << "rotAngDEG: " << (result * Utils::RAD2DEG) << std::endl;
    }

    return result;
}

SLVec2f CompassAlignment::calculateExpectedTemplatePixelLocation(const SLDeviceLocation& devLoc,
                                                                 const SENSCalibration&  calibration,
                                                                 SLVec3f                 templateLatLonDegAltM,
                                                                 SLMat4f                 camPose)
{
    SLVec3f wTc = camPose.translation(); // camera w.r.t. world

    SLVec3d tplLocEcef;
    tplLocEcef.latlonAlt2ecef(SLVec3d(templateLatLonDegAltM.x, templateLatLonDegAltM.y, templateLatLonDegAltM.z));
    SLVec3d enuTtpl = devLoc.wRecef() * tplLocEcef;
    SLVec3d wTtpl   = enuTtpl - devLoc.originENU(); // tpl location w.r.t. world
    SLVec3f wTtpl_f = SLVec3f(wTtpl.x, wTtpl.y, wTtpl.z);
    SLVec3f cTtpl   = camPose.mat3() * (wTtpl_f - wTc); // tpl location w.r.t. camera

    SLMat3f camMat = SLMat3f(calibration.fx(),
                             0.0f,
                             calibration.cx(),
                             0.0f,
                             calibration.fy(),
                             calibration.cy(),
                             0.0f,
                             0.0f,
                             1.0f);
    SLVec3f fTtpl  = camMat * cTtpl;                                // tpl location w.r.t. film
    SLVec2f pTtpl  = SLVec2f(fTtpl.x / fTtpl.z, fTtpl.y / fTtpl.z); // expected pixel location of projected template

    return pTtpl;
}
