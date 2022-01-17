//#############################################################################
//  File:      SpryTrackCalibrator.cpp
//  Date:      January 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SpryTrackCalibrator.h>
#include <utility>

//-----------------------------------------------------------------------------
SpryTrackCalibrator::SpryTrackCalibrator(SpryTrackDevice& device,
                                         CVSize2f         planeSize)
  : _device(device), _planeSize(std::move(planeSize))
{
}
//-----------------------------------------------------------------------------
void SpryTrackCalibrator::calibrate()
{
    // Create and register the calibration marker
    auto* marker = _device.markers()[0];
    //    marker->addPoint(0.0f, 11.0f, 3.0f);
    //    marker->addPoint(-15.0f, -26.0f, 3.0f);
    //    marker->addPoint(16.59f, -20.95f, 3.0f);
    //    marker->addPoint(0.0f, 0.0f, 0.0f);
    //    marker->addPoint(_planeSize.width, 0.0f, 0.0f);
    //    marker->addPoint(0.0f, 0.0f, _planeSize.height);
    //    marker->addPoint(_planeSize.width, 0.0f, _planeSize.height);
    //    _device.registerMarker(marker);

    // Acquire and process the next frame
    _device.acquireFrame();

    // Unregister the calibration marker
    //    _device.unregisterMarker(marker);

    if (marker->visible())
    {
        // Invert the marker object view matrix to get the extrinsic calibration matrix
        _extrinsicMat = marker->objectViewMat();
    }
    else
    {
        throw SpryTrackCalibrationException("SpryTrack: Calibration error (marker was not detected)");
    }
}
//-----------------------------------------------------------------------------