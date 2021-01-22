#ifndef COMPASS_ALIGNMENT_H
#define COMPASS_ALIGNMENT_H

#include <opencv2/core.hpp>

#include <SLVec3.h>
#include <SLMat4.h>
#include <SLDeviceLocation.h>

#include <SENSCalibration.h>
#include <SENSFrame.h>

class CompassAlignment
{
public:
    struct Template
    {
        cv::Mat image;
        SLVec3f latDegLonDegAltM;
    };

    static float   calculateRotAngle(const SLDeviceLocation& devLoc,
                                     const SENSCalibration&  calibration,
                                     Template&               tpl,
                                     SLMat4f&                camPose,
                                     SENSFramePtr&           frame);
    static SLVec2f calculateExpectedTemplatePixelLocation(const SLDeviceLocation& devLoc,
                                                          const SENSCalibration&  calibration,
                                                          SLVec3f                 templateLatLonDegAltM,
                                                          SLMat4f                 camPose);

    static Template createTemplate(const std::string& imagePath, float latDeg, float lonDeg, float altM)
    {
        Template result;

        result.image            = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        result.latDegLonDegAltM = SLVec3f(latDeg, lonDeg, altM);

        return result;
    }
};

#endif