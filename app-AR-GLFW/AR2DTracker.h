//#############################################################################
//  File:      AR2DTracker.h
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
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
                AR2DTracker     (cv::Mat intrinsics,
                                 cv::Mat distoriton);

        bool    init            (string paramsFileDir) override;
        bool    track           () override;
        void    updateSceneView (ARSceneView* sv) override;
        void    unloadSGObjects () override;

    private:
        AR2DMap _map;
        SLNode* _node;

        cv::Ptr<cv::FeatureDetector>    _detector;
        cv::Ptr<cv::DescriptorMatcher>  _matcher;

        vector<cv::KeyPoint> _sceneKeypoints;
        cv::Mat             _sceneDescriptors;

        bool                _posInitialized;//true, if we have a valid pos. and can use optical flow
        bool                _posValid;      //last call of track() calculated a valid position

        vector<cv::Point2f> _scenePts;
        vector<cv::Point3f> _mapPts;
};
//-----------------------------------------------------------------------------

#endif // AR2DTRACKER_H
