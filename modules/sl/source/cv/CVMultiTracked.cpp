//#############################################################################
//  File:      CVMultiTracked.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <cv/CVMultiTracked.h>
#include <vector>

//-----------------------------------------------------------------------------
CVMultiTracked::CVMultiTracked(CVTracked* tracked)
  : _tracked(tracked)
{
}
//-----------------------------------------------------------------------------
CVMultiTracked::~CVMultiTracked() noexcept
{
    delete _tracked;
}
//-----------------------------------------------------------------------------
/*! Forwards the call to the underlying CVTracked* and stores the result in
 * the internal matrix vector for later averaging
 * @param imageGray
 * @param imageRgb
 * @param calib
 * @return
 */
bool CVMultiTracked::track(CVMat          imageGray,
                           CVMat          imageRgb,
                           CVCalibration* calib)
{
    bool result = _tracked->track(imageGray, imageRgb, calib);

    if (result)
    {
        _objectViewMatrices.push_back(_tracked->objectViewMat());
    }

    return result;
}
//-----------------------------------------------------------------------------
/*! Averages the object view matrices computed before using the "track" method
 * and clears the internal matrix vector
 */
void CVMultiTracked::combine()
{
    // Early exit if there's only one matrix
    if (_objectViewMatrices.size() == 1)
    {
        _objectViewMat = CVMatx44f(_objectViewMatrices[0].val);
        _objectViewMatrices.clear();
        return;
    }

    std::vector<CVVec3f>  positions;
    std::vector<SLQuat4f> rotations;
    std::vector<float>    weights;

    for (CVMatx44f& matrix : _objectViewMatrices)
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

    CVVec3f  avgPosition = averageVector(positions, weights);
    SLQuat4f avgRotation = averageQuaternion(rotations, weights);
    SLMat3f  avgRotMat   = avgRotation.toMat3();

    // clang-format off
    _objectViewMat = CVMatx44f(avgRotMat.m(0), avgRotMat.m(3), avgRotMat.m(6), avgPosition.val[0],
                               avgRotMat.m(1), avgRotMat.m(4), avgRotMat.m(7), avgPosition.val[1],
                               avgRotMat.m(2), avgRotMat.m(5), avgRotMat.m(8), avgPosition.val[2],
                               0, 0, 0, 1);
    // clang-format on

    _objectViewMatrices.clear();
}
//-----------------------------------------------------------------------------