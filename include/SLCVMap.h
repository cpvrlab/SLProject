//#############################################################################
//  File:      SLCVMap.h
//  Author:    Michael Goettlicher
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
class SLCVKeyFrameDB;
//class SLCVVKeyFrame;

using namespace std;

//-----------------------------------------------------------------------------
//! 
/*! 
*/
class SLCVMap
{
public:
    SLCVMap(const string& name);
    ~SLCVMap();

    //! get reference to map points vector
    std::vector<SLCVMapPoint*>& mapPoints() { return _mapPoints; }
    SLCVKeyFrameDB* getKeyFrameDB() { return mpKeyFrameDatabase; }

    //! get visual representation as SLPoints
    SLPoints* getSceneObject();
    SLPoints* getNewSceneObject();
    void SetReferenceMapPoints(const std::vector<SLCVMapPoint*> &vpMPs);

    void setKeyFrameDB(SLCVKeyFrameDB* kfDB) { mpKeyFrameDatabase = kfDB; }
    void clear();
    long unsigned int KeyFramesInMap();

    vector<SLCVKeyFrame*> mvpKeyFrameOrigins;
private:
    //SLCVVMapPoint _mapPoints;
    std::vector<SLCVMapPoint*> _mapPoints;

    std::vector<SLCVMapPoint*> mvpReferenceMapPoints;

    //Pointer to visual representation object (ATTENTION: do not delete this object)
    SLPoints* _sceneObject = NULL;

    SLCVKeyFrameDB* mpKeyFrameDatabase=NULL;
};

#endif // !SLCVMAP_H
