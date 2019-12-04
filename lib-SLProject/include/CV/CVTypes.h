//#############################################################################
//  File:      CVTypes.h
//  Author:    Michael Goettlicher
//  Date:      Winter 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVTYPES_H
#define CVTYPES_H

class CVCalibrationEstimatorParams
{
public:
    void toggleFixPrincipalPoint() { fixPrincipalPoint = !fixPrincipalPoint; }
    void toggleFixAspectRatio() { fixAspectRatio = !fixAspectRatio; }
    void toggleZeroTangentDist() { zeroTangentDistortion = !zeroTangentDistortion; }
    void toggleRationalModel() { calibRationalModel = !calibRationalModel; }
    void toggleTiltedModel() { calibTiltedModel = !calibTiltedModel; }
    void toggleThinPrismModel() { calibThinPrismModel = !calibThinPrismModel; }

    bool fixPrincipalPoint     = false;
    bool fixAspectRatio        = false;
    bool zeroTangentDistortion = false;
    bool calibRationalModel    = false;
    bool calibTiltedModel      = false;
    bool calibThinPrismModel   = false;
};

enum class CVCameraType
{
    FRONTFACING,
    BACKFACING,
    VIDEOFILE
};

#endif // CVCALIBRATIONESTIMATORPARAMS_H
