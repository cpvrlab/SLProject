//#############################################################################
//  File:      SpryTrackCalibrator.cpp
//  Date:      January 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SpryTrackCalibrator.h>

//-----------------------------------------------------------------------------
SpryTrackCalibrator::SpryTrackCalibrator(SpryTrackDevice& device)
  : _device(device) {}
//-----------------------------------------------------------------------------
void SpryTrackCalibrator::calibrate()
{
    // Create and register the calibration marker
    auto* marker = new SpryTrackMarker();
    // TODO: Figure out points
    _device.registerMarker(marker);

    // Acquire the next frame
    int      width;
    int      height;
    uint8_t* dataGrayLeft;
    uint8_t* dataGrayRight;
    _device.acquireFrame(&width, &height, &dataGrayLeft, &dataGrayRight);

    // Invert the marker object view matrix to get the extrinsic calibration matrix
    CVMatx44f camToWorldMat = marker->objectViewMat();
    _extrinsicMat           = camToWorldMat.inv();

    // Unregister the calibration marker
    _device.unregisterMarker(marker);
}
//-----------------------------------------------------------------------------