//#############################################################################
//  File:      SpryTrackMarker.h
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_SPRYTRACKMARKER_H
#define SRC_SPRYTRACKMARKER_H

#include <ftkInterface.h>
#include <SL.h>
#include <SLMat4.h>

//-----------------------------------------------------------------------------
typedef uint32 SpryTrackMarkerID;
//-----------------------------------------------------------------------------
class SpryTrackMarker
{
    friend class SpryTrackDevice;

public:
    SpryTrackMarkerID id() const { return _geometry.geometryId; }
    SLMat4f           objectViewMat() const { return _objectViewMat; }
    SLbool            visible() const { return _visible; }

    void addPoint(float x, float y, float z);

private:
    void update(ftkMarker& marker);

    ftkGeometry _geometry;
    SLMat4f     _objectViewMat;
    SLbool      _visible = false;
};
//-----------------------------------------------------------------------------

#endif // SRC_SPRYTRACKMARKER_H
