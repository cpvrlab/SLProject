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

//-----------------------------------------------------------------------------
//! 
/*!
*/
class SLCVSlamStateLoader
{
public:
    //! Opens and parses file with opencvs FileStorage
    SLCVSlamStateLoader(const string& filename);
    ~SLCVSlamStateLoader();
    //! execute loading procedure
    void load();

protected:
    
private:
    void loadKeyFrames();
    void loadMapPoints();

    std::vector<SLCVMapPoint> _mapPts;
    std::vector<SLCVKeyFrame> _keyFrames;

    cv::FileStorage _fs;
};

#endif // !SLCV_SLAMSTATELOADER_H
