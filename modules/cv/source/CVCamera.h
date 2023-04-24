//#############################################################################
//  File:      CVCalibration.h
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVCAMERA_H
#define CVCAMERA_H

#include <CVTypes.h>
#include <CVCalibration.h>

//-----------------------------------------------------------------------------
class CVCamera
{
public:
    CVCamera(CVCameraType type);

    bool         mirrorH() { return _mirrorH; }
    bool         mirrorV() { return _mirrorV; }
    CVCameraType type() { return _type; }
    void         showUndistorted(bool su) { _showUndistorted = su; }
    bool         showUndistorted() { return _showUndistorted; }
    int          camSizeIndex() { return _camSizeIndex; }

    void camSizeIndex(int index)
    {
        _camSizeIndex = index;
    }
    void toggleMirrorH() { _mirrorH = !_mirrorH; }
    void toggleMirrorV() { _mirrorV = !_mirrorV; }

    CVCalibration calibration;

private:
    bool         _showUndistorted = false; //!< Flag if image should be undistorted
    CVCameraType _type;
    bool         _mirrorH = false;
    bool         _mirrorV = false;

    int _camSizeIndex = -1;
};
//-----------------------------------------------------------------------------
#endif // CVCAMERA_H
