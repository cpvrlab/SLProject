//#############################################################################
//  File:      SLCVMapPoint.h
//  Author:    Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVMAPPOINT_H
#define SLCVMAPPOINT_H

#include <vector>
#include <SLCV.h>
//-----------------------------------------------------------------------------
//! 
/*! 
*/
class SLCVMapPoint
{
public:
    void id(int id) { _id = id; }
    void worldPos(const SLCVMat& pos) { _worldPos = pos; }
    SLVec3f worldPos();
private:
    int _id;
    //open cv coordinate representation: z-axis points to principlal point,
    // x-axis to the right and y-axis down
    SLCVMat _worldPos;
};

typedef std::vector<SLCVMapPoint> SLCVVMapPoint;

#endif // !SLCVMAPPOINT_H
