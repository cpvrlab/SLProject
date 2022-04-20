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
#include <stdexcept>

//-----------------------------------------------------------------------------
constexpr int MAX_FAILED_ACQUISITION_ATTEMPTS = 100;
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
    float squareSize = 60.0f;

    auto* marker = new SpryTrackMarker(200);
    /*marker->addPoint(2 * squareSize, 0.0f, 4 * squareSize);
    marker->addPoint(4 * squareSize, 0.0f, 6 * squareSize);
    marker->addPoint(6 * squareSize, 0.0f, 7 * squareSize);
    marker->addPoint(5 * squareSize, 0.0f, 2 * squareSize);
    marker->addPoint(1 * squareSize, 0.0f, 8 * squareSize);*/

    marker->addPoint(0 * squareSize, 0.0f, 1 * squareSize);
    marker->addPoint(1 * squareSize, 0.0f, 3 * squareSize);
    marker->addPoint(2 * squareSize, 0.0f, 2 * squareSize);
    marker->addPoint(3 * squareSize, 0.0f, 0 * squareSize);
    marker->addPoint(4 * squareSize, 0.0f, 4 * squareSize);
    _device.registerMarker(marker);

    for (int i = 0; i < MAX_FAILED_ACQUISITION_ATTEMPTS && !marker->visible(); i++)
    {
        // Acquire and process the next frame
        _device.acquireFrame();
    }

    if (marker->visible())
    {
        // Invert the marker object view matrix to get the extrinsic calibration matrix
        _extrinsicMat = marker->objectViewMat();
        _device.unregisterMarker(marker);
    }
    else
    {
        _device.unregisterMarker(marker);
        throw std::runtime_error("Calibration error (marker was not detected)");
    }
}
//-----------------------------------------------------------------------------