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
class SLCVVKeyFrame;
class SLCVMapNode;

using namespace std;

//-----------------------------------------------------------------------------
//! 
/*! 
*/
class SLCVMap
{
public:
    enum TransformType {
        ROT_X = 0, ROT_Y, ROT_Z, TRANS_X, TRANS_Y, TRANS_Z, SCALE
    };

    SLCVMap(const string& name);
    ~SLCVMap();

    //! get reference to map points vector
    //std::vector<SLCVMapPoint*>& mapPoints() { return _mapPoints; }
    std::vector<SLCVMapPoint*> GetAllMapPoints();
    const std::set<SLCVMapPoint*>& GetAllMapPointsConstRef() const;
    std::set<SLCVMapPoint*>& GetAllMapPointsRef();
    void AddMapPoint(SLCVMapPoint *pMP);
    SLCVKeyFrameDB* getKeyFrameDB() { return mpKeyFrameDatabase; }
    void AddKeyFrame(SLCVKeyFrame* pKF);

    std::vector<SLCVKeyFrame*> GetAllKeyFrames();

    //! get visual representation as SLPoints
    //SLPoints* getSceneObject();
    //SLPoints* getNewSceneObject();
    void SetReferenceMapPoints(const std::vector<SLCVMapPoint*> &vpMPs);

    long unsigned int MapPointsInMap();

    void setKeyFrameDB(SLCVKeyFrameDB* kfDB) { mpKeyFrameDatabase = kfDB; }
    void clear();
    long unsigned int KeyFramesInMap();

    void EraseMapPoint(SLCVMapPoint *pMP);
    void EraseKeyFrame(SLCVKeyFrame* pKF);

    vector<SLCVKeyFrame*> mvpKeyFrameOrigins;

    //set map node for visu update
    void setMapNode(SLCVMapNode* mapNode);

    //transformation functions
    void rotate(float value, int type);
    void translate(float value, int type);
    void scale(float value);
    void applyTransformation(double value, TransformType type);
    cv::Mat buildTransMat(float &val, int type);
    cv::Mat buildRotMat(float &valDeg, int type);
private:
    //SLCVVMapPoint _mapPoints;
    //std::vector<SLCVMapPoint*> _mapPoints;
    long unsigned int mnMaxKFid;

    std::set<SLCVMapPoint*> mspMapPoints;
    std::set<SLCVKeyFrame*> mspKeyFrames;

    std::vector<SLCVMapPoint*> mvpReferenceMapPoints;

    //Pointer to visual representation object (ATTENTION: do not delete this object)
    SLPoints* _sceneObject = NULL;

    SLCVKeyFrameDB* mpKeyFrameDatabase=NULL;

    SLCVMapNode* _mapNode = NULL;
};

#endif // !SLCVMAP_H
