//#############################################################################
//  File:      AR2DTracker.h
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef AR2DTRACKER_H
#define AR2DTRACKER_H

#include <SLNode.h>
#include <ARTracker.h>
#include <AR2DMapper.h>
#include <opencv2/features2d.hpp>

//-----------------------------------------------------------------------------
class AR2DTracker : public ARTracker
{
public:
    AR2DTracker(cv::Mat intrinsics, cv::Mat distoriton);

    bool init(string paramsFileDir) override;
    bool track() override;
    void updateSceneView( ARSceneView* sv ) override;
    void unloadSGObjects() override;

private:
    AR2DMap _map;

    SLNode* _node;

    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorMatcher> _matcher;

     std::vector<cv::KeyPoint> _sceneKeypoints;
     cv::Mat _sceneDescriptors;

     //true, if we currently have a valid position and can use optical flow
     bool _posInitialized;
     //last call of track() calculated a valid position
     bool _posValid;

     std::vector<cv::Point2f> _scenePts;
     std::vector<cv::Point3f> _mapPts;
};
//-----------------------------------------------------------------------------

#endif // AR2DTRACKER_H
