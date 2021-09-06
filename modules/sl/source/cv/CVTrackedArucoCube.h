//#############################################################################
//  File:      CVTrackedArucoCube.h
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_CVTRACKEDARUCOCUBE_H
#define SLPROJECT_CVTRACKEDARUCOCUBE_H

#include <cv/CVTypedefs.h>
#include <cv/CVTrackedAruco.h>

enum CVTrackedArucoCubeFace
{
    ACF_left   = 0,
    ACF_right  = 1,
    ACF_bottom = 2,
    ACF_top    = 3,
    ACF_back   = 4,
    ACF_front  = 5
};

class CVTrackedArucoCube : public CVTrackedAruco
{
public:
    static const int NO_MARKER = -1;

    CVTrackedArucoCube(const int trackedMarkerIDs[6], string calibIniPath);

    bool track(CVMat          imageGray,
               CVMat          imageRgb,
               CVCalibration* calib);

private:
    CVMatx44f getFaceToCubeRotation(CVTrackedArucoCubeFace face);

private:
    int    _trackedMarkerIDs[6];
    string _calibIniPath;
};

#endif // SLPROJECT_CVTRACKEDARUCOCUBE_H
