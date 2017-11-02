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

class SLPoints;

using namespace std;

//-----------------------------------------------------------------------------
//! 
/*! 
*/
class SLCVMap
{
public:
    SLCVMap(const string& name);

    //! get reference to map points vector
    SLCVVMapPoint& mapPoints() { return _mapPoints; }

    //! get visual representation as SLPoints
    SLPoints* getSceneObject();
protected:


private:
    SLCVVMapPoint _mapPoints;
    //Pointer to visual representation object (ATTENTION: do not delete this object)
    SLPoints* _sceneObject = NULL;
};

#endif // !SLCVMAP_H
