//#############################################################################
//  File:      SpryTrackMarker.cpp
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SpryTrackMarker.h>

//-----------------------------------------------------------------------------
constexpr float MM_TO_M = 1.0f / 1000.0f;
//-----------------------------------------------------------------------------
void SpryTrackMarker::addPoint(float x, float y, float z)
{
    if (_geometry.pointsCount == 6u)
    {
        SL_EXIT_MSG("SpryTrack: Cannot add more than 6 points to a single geometry");
    }

    uint32 index                 = _geometry.pointsCount;
    _geometry.positions[index].x = x;
    _geometry.positions[index].y = y;
    _geometry.positions[index].z = z;
    _geometry.pointsCount++;
}
//-----------------------------------------------------------------------------
void SpryTrackMarker::update(ftkMarker& marker)
{
    // Column 1
    _objectViewMat.m( 0, marker.rotation[0][0]);
    _objectViewMat.m( 1, marker.rotation[1][0]);
    _objectViewMat.m( 2, marker.rotation[2][0]);
    _objectViewMat.m( 3, 0.0);

    // Column 2
    _objectViewMat.m( 4, marker.rotation[0][1]);
    _objectViewMat.m( 5, marker.rotation[1][1]);
    _objectViewMat.m( 6, marker.rotation[2][1]);
    _objectViewMat.m( 7, 0.0);

    // Column 3
    _objectViewMat.m( 8, marker.rotation[0][2]);
    _objectViewMat.m( 9, marker.rotation[1][2]);
    _objectViewMat.m(10, marker.rotation[2][2]);
    _objectViewMat.m(11, 0.0);

    // Column 4
    _objectViewMat.m(12, marker.translationMM[0] * MM_TO_M);
    _objectViewMat.m(13, marker.translationMM[1] * MM_TO_M);
    _objectViewMat.m(14, marker.translationMM[2] * MM_TO_M);
    _objectViewMat.m(15, 1.0);
}