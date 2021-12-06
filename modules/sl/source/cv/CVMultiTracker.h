//#############################################################################
//  File:      CVMultiTracker.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_CVMULTITRACKER_H
#define SLPROJECT_CVMULTITRACKER_H

#include <cv/CVTracked.h>
#include <cv/CVTypedefs.h>

#include <Averaged.h>
#include <cv/CVTrackedArucoCube.h>

#include <SLQuat4.h>

#include <vector>

//-----------------------------------------------------------------------------
//! CVMultiTracker is used for tracking the same object in multiple frames
/*! The CVMultiTracker class averages the object view matrices of the same
 * object in multiple frames.
 * The class is given a pointer to a CVTracked which is used for tracking in
 * a single frame of data. The "track" method can then be called an arbitrary
 * number of times. Every call, the "track" method of the provided CVTracked*
 * is called and the resulting object view matrix stored in a vector.
 * To get the averaged object view matrix of the CVMultiTracker, the method
 * "combine" can called. This also clears the internal vector for new frames.
 */
class CVMultiTracker
{
private:
    CVVMatx44f         _worldMatrices;
    std::vector<float> _weights;
    CVMatx44f          _averageWorldMatrix;

    Averaged<CVVec3f> _averagePosition = Averaged<CVVec3f>(6, CVVec3f(0.0f, 0.0f, 0.0f));
    AveragedQuat4f    _averageRotation = AveragedQuat4f(6, SLQuat4f(0.0f, 0.0f, 0.0f, 1.0f));

public:
    void      recordCurrentPose(CVTracked* tracked, CVCalibration* calib);
    void      combine();
    CVMatx44f averageWorldMatrix() { return _averageWorldMatrix; }
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_CVMULTITRACKER_H
