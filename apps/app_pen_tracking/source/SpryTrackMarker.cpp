//#############################################################################
//  File:      SpryTrackMarker.cpp
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SpryTrackMarker.h>
#include <CoordSystemConversions.h>

//-----------------------------------------------------------------------------
using namespace CoordSystemConversions;
//-----------------------------------------------------------------------------
constexpr float MM_TO_M = 1.0f / 1000.0f;
//-----------------------------------------------------------------------------
void SpryTrackMarker::addPoint(float x, float y, float z)
{
    if (_geometry.pointsCount == FTK_MAX_FIDUCIALS)
    {
        SL_EXIT_MSG(("SpryTrack: Cannot add more than " +
                     std::to_string(FTK_MAX_FIDUCIALS) +
                     " points to a single geometry")
                      .c_str());
    }

    float position[] = {x, y, z};
    convert3<POS_X, POS_Y, NEG_Z>(position, position);

    uint32 index                 = _geometry.pointsCount;
    _geometry.positions[index].x = position[0];
    _geometry.positions[index].y = position[1];
    _geometry.positions[index].z = position[2];
    _geometry.pointsCount++;
}
//-----------------------------------------------------------------------------
void SpryTrackMarker::update(ftkMarker& marker)
{
    // Column 1
    _objectViewMat(0, 0) = marker.rotation[0][0];
    _objectViewMat(1, 0) = marker.rotation[1][0];
    _objectViewMat(2, 0) = marker.rotation[2][0];
    _objectViewMat(3, 0) = 0.0;

    // Column 2
    _objectViewMat(0, 1) = marker.rotation[0][1];
    _objectViewMat(1, 1) = marker.rotation[1][1];
    _objectViewMat(2, 1) = marker.rotation[2][1];
    _objectViewMat(3, 1) = 0.0;

    // Column 3
    _objectViewMat(0, 2) = marker.rotation[0][2];
    _objectViewMat(1, 2) = marker.rotation[1][2];
    _objectViewMat(2, 2) = marker.rotation[2][2];
    _objectViewMat(3, 2) = 0.0;

    // Column 4
    _objectViewMat(0, 3) = marker.translationMM[0] * MM_TO_M;
    _objectViewMat(1, 3) = marker.translationMM[1] * MM_TO_M;
    _objectViewMat(2, 3) = marker.translationMM[2] * MM_TO_M;
    _objectViewMat(3, 3) = 1.0;

    // Convert coordinate system
    convert4x4<POS_X, NEG_Y, NEG_Z, false, false>(_objectViewMat.val,
                                                  _objectViewMat.val);
}
//-----------------------------------------------------------------------------