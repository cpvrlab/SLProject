//#############################################################################
//  File:      SLCVSlamStateLoader.h
//  Author:    Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCV_SLAMSTATELOADER_H
#define SLCV_SLAMSTATELOADER_H

#include <vector>
#include <opencv2/core/core.hpp>

#include <SLCVKeyFrame.h>
#include <OrbSlam/ORBVocabulary.h>

class SLCVKeyFrameDB;
class SLCVMap;

using namespace ORB_SLAM2;
//-----------------------------------------------------------------------------
//! 
/*!
*/
class SLCVSlamStateLoader
{
public:
    //! Opens and parses file with opencvs FileStorage
    SLCVSlamStateLoader(const string& filename, ORBVocabulary* orbVoc, bool loadKfImgs=true);
    ~SLCVSlamStateLoader();
    //! execute loading procedure
    void load(SLCVMap& map, SLCVKeyFrameDB& kfDB);

protected:
    
private:
    void loadKeyFrames(SLCVMap& map, SLCVKeyFrameDB& kfDB);
    void loadMapPoints(SLCVMap& map);
    //calculation of scaleFactors , levelsigma2, invScaleFactors and invLevelSigma2
    void calculateScaleFactors(float scaleFactor, int nlevels);

    cv::FileStorage _fs;
    ORBVocabulary* _orbVoc;

    //load keyframe images
    bool _loadKfImgs = false;

    //mapping of keyframe pointer by their id (used during map points loading)
    map<int, SLCVKeyFrame*> _kfsMap;

    float _s=200.f;
    cv::Mat _t;
    cv::Mat _rot;

    //vectors for precalculation of scalefactors
    std::vector<float> _vScaleFactor;
    std::vector<float> _vInvScaleFactor;
    std::vector<float> _vLevelSigma2;
    std::vector<float> _vInvLevelSigma2;
};

#endif // !SLCV_SLAMSTATELOADER_H
