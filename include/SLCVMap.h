//#############################################################################
//  File:      SLCVMap.h
//  Author:    Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVMAP_H
#define SLCVMAP_H

#include <vector>
#include <string>
#include <SLCVMapPoint.h>

using namespace std;

//-----------------------------------------------------------------------------
//! 
/*! 
*/
class SLCVMap
{
public:
    SLCVMap(const string& name);
    //! add map point
    void addPoint(const SLCVMapPoint& mapPt);

    //! get visual representation as SLPoints
protected:


private:
    vector<SLCVMapPoint> _mapPoints;

};

#endif // !SLCVMAP_H
