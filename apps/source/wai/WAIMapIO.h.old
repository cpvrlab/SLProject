//#############################################################################
//  File:      WAIMapIO.h
//  Author:    Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef WAI_MAPIO_H
#define WAI_MAPIO_H

#include <vector>
#include <opencv2/core/core.hpp>

#include <WAIMap.h>
#include <WAIKeyFrameDB.h>
#include <WAIKeyFrame.h>
#include <OrbSlam/ORBVocabulary.h>

#include <Utils.h>

using namespace ORB_SLAM2;
//-----------------------------------------------------------------------------
//!
/*!
*/
class WAIMapIO
{
    public:
    //! Opens and parses file with opencvs FileStorage
    WAIMapIO(const string& filename, ORBVocabulary* orbVoc, bool kfImgsIO = true, std::string currImgPath = "");
    ~WAIMapIO();
    //! execute loading procedure
    void        load(cv::Mat& om, WAIMap& map, WAIKeyFrameDB& kfDB);
    static void save(const string& filename, WAIMap& map, bool kfImgsIO, const string& pathImgs, cv::Mat om);

    protected:
    private:
    void loadKeyFrames(WAIMap& map, WAIKeyFrameDB& kfDB);
    void loadMapPoints(WAIMap& map);
    //calculation of scaleFactors , levelsigma2, invScaleFactors and invLevelSigma2
    void calculateScaleFactors(float scaleFactor, int nlevels);

    cv::FileStorage _fs;
    ORBVocabulary*  _orbVoc;

    //load keyframe images
    bool        _kfImgsIO = false;
    std::string _currImgPath;

    //mapping of keyframe pointer by their id (used during map points loading)
    map<int, WAIKeyFrame*> _kfsMap;

    float   _s = 200.f;
    cv::Mat _t;
    cv::Mat _rot;

    //vectors for precalculation of scalefactors
    std::vector<float> _vScaleFactor;
    std::vector<float> _vInvScaleFactor;
    std::vector<float> _vLevelSigma2;
    std::vector<float> _vInvLevelSigma2;
};

#endif // !WAI_MAPIO_H
