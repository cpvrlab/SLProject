//#############################################################################
//  File:      CVMultiTracker.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <cv/CVMultiTracker.h>
#include <utility>
#include <vector>

//-----------------------------------------------------------------------------
/*! Computes the matrix in world space for the current object view matrix of
 * the tracker and saves it for later averaging
 */
void CVMultiTracker::recordCurrentPose(CVTracked* tracked, CVCalibration* calib)
{
    CVMatx44f matrix(tracked->objectViewMat());

    if (!calib->rvec.empty() && !calib->tvec.empty())
    {
        CVMatx44f extrinsic = CVTracked::createGLMatrix(calib->tvec, calib->rvec);
        // clang-format off
        extrinsic = CVMatx44f(extrinsic.val[ 1], extrinsic.val[ 2], extrinsic.val[ 0], extrinsic.val[3],
                              extrinsic.val[ 5], extrinsic.val[ 6], extrinsic.val[ 4], extrinsic.val[7],
                              extrinsic.val[ 9], extrinsic.val[10], extrinsic.val[ 8], extrinsic.val[11],
                              0.0f,              0.0f,             0.0f,               1.0f);
        // clang-format on
        matrix = extrinsic.inv() * matrix;
    }
    else
    {
        //        SL_WARN_MSG("Camera extrinsic calibration not available -> multi tracked matrix will be in camera space");
    }

    _worldMatrices.push_back(matrix);
}
//-----------------------------------------------------------------------------
/*! Averages the object view matrices computed before using the "track" method
 * and clears the internal matrix vector
 */
void CVMultiTracker::combine()
{
    if (_worldMatrices.empty())
    {
        return;
    }

    std::vector<CVVec3f>  positions;
    std::vector<SLQuat4f> rotations;
    std::vector<float>    weights;

    for (CVMatx44f& matrix : _worldMatrices)
    {
        positions.emplace_back(matrix.val[3], matrix.val[7], matrix.val[11]);

        // clang-format off
        SLMat3f rotMat(matrix.val[0], matrix.val[1], matrix.val[2],
                       matrix.val[4], matrix.val[5], matrix.val[6],
                       matrix.val[8], matrix.val[9], matrix.val[10]);
        // clang-format on
        SLQuat4f rot;
        rot.fromMat3(rotMat);
        rotations.push_back(rot);

        weights.push_back(1.0f);
    }

    CVVec3f  avgPosition = CVTracked::averageVector(positions, weights);
    SLQuat4f avgRotation = CVTracked::averageQuaternion(rotations, weights);
    SLMat3f  avgRotMat   = avgRotation.toMat3();

    // clang-format off
    _averageWorldMatrix = CVMatx44f(avgRotMat.m(0), avgRotMat.m(3), avgRotMat.m(6), avgPosition.val[0],
                                    avgRotMat.m(1), avgRotMat.m(4), avgRotMat.m(7), avgPosition.val[1],
                                    avgRotMat.m(2), avgRotMat.m(5), avgRotMat.m(8), avgPosition.val[2],
                                    0, 0, 0, 1);
    // clang-format on

    _worldMatrices.clear();
}
//-----------------------------------------------------------------------------