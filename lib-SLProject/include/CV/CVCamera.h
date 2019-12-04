//#############################################################################
//  File:      CVCalibration.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVCAMERA_H
#define CVCAMERA_H

#include <CVTypes.h>
#include <CVCalibration.h>

class CVCamera
{
public:
    CVCamera(CVCameraType type);
    void toggleMirrorH() { _mirrorH = !_mirrorH; }
    void toggleMirrorV() { _mirrorV = !_mirrorV; }

private:
    CVCameraType _type;
    bool         _mirrorH = false;
    bool         _mirrorV = false;

    CVCalibration _calibration;
};

#endif // CVCAMERA_H
