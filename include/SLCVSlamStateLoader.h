//#############################################################################
//  File:      SLCVSlamStateLoader.h
//  Author:    Michael Göttlicher
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

#include <SLCVMapPoint.h>
#include <SLCVKeyFrame.h>
#include <OrbSlam\ORBVocabulary.h>

using namespace ORB_SLAM2;
//-----------------------------------------------------------------------------
//! 
/*!
*/
class SLCVSlamStateLoader
{
public:
    //! Opens and parses file with opencvs FileStorage
    SLCVSlamStateLoader(const string& filename, ORBVocabulary* orbVoc);
    ~SLCVSlamStateLoader();
    //! execute loading procedure
    void load( SLCVVMapPoint& mapPts, SLCVVKeyFrame& kfs);

protected:
    
private:
    void loadKeyFrames( SLCVVKeyFrame& kfs );
    void loadMapPoints( SLCVVMapPoint& mapPts );

    cv::FileStorage _fs;
    ORBVocabulary* _orbVoc;

    //mapping of keyframe pointer by their id (used during map points loading)
    map<int, SLCVKeyFrame*> _kfsMap;
};

#endif // !SLCV_SLAMSTATELOADER_H
