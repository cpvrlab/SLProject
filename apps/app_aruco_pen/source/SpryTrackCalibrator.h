//#############################################################################
//  File:      SpryTrackCalibrator.h
//  Date:      January 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_SPRYTRACKCALIBRATOR_H
#define SRC_SPRYTRACKCALIBRATOR_H

#include <SpryTrackDevice.h>
#include <CVTypedefs.h>
#include <stdexcept>

//-----------------------------------------------------------------------------
class SpryTrackCalibrationException : public std::runtime_error
{
public:
    explicit SpryTrackCalibrationException(const string& message)
      : std::runtime_error(message)
    {
    }
};
//-----------------------------------------------------------------------------
class SpryTrackCalibrator
{
public:
    SpryTrackCalibrator(SpryTrackDevice& device,
                        CVSize2f         planeSize);
    void calibrate();

    // Getters
    CVMatx44f extrinsicMat() { return _extrinsicMat; }

private:
    SpryTrackDevice& _device;
    CVSize2f         _planeSize;

    CVMatx44f _extrinsicMat;
};
//-----------------------------------------------------------------------------
#endif // SRC_SPRYTRACKCALIBRATOR_H
