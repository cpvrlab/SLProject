//#############################################################################
//  File:      SLTrackingInfosInterface.h
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_TRACKINGINFOSINTERFACE_H
#define SL_TRACKINGINFOSINTERFACE_H

#include <string>

class SLTrackingInfosInterface
{
public:
    virtual ~SLTrackingInfosInterface() {}
    //!get current tracking state
    virtual string getPrintableState() = 0;
    //!get current tracking type
    virtual string getPrintableType() = 0;
    //!get mean reprojection error
    virtual float meanReprojectionError() = 0;
    //!get number of matches in current frame to the slam map
    virtual int getNMapMatches() = 0;
    //!get camera pose difference to previous camera pose
    virtual float poseDifference() = 0;
    //!get number of map points
    virtual int mapPointsCount() = 0;
    //!get number of key frames
    virtual int getNumKeyFrames() = 0;

    //!getters
    bool showKeyPoints() const { return _showKeyPoints; }
    bool showKeyPointsMatched() const { return _showKeyPointsMatched; }
    bool showMapPC() const { return _showMapPC; }
    bool showMatchesPC() const { return _showMatchesPC; }
    bool showLocalMapPC() const { return _showLocalMapPC; }
    bool showKeyFrames() const { return _showKeyFrames; }
    bool showCovisibilityGraph() const { return _showCovisibilityGraph; }
    bool showSpanningTree() const { return _showSpanningTree; }
    bool showLoopEdges() const { return _showLoopEdges; }
    bool renderKfBackground() const { return _renderKfBackground; }
    bool allowKfsAsActiveCam() const { return _allowKfsAsActiveCam; }

    //!setters
    void showKeyPoints(bool state) { _showKeyPoints = state; }
    void showKeyPointsMatched(bool state) { _showKeyPointsMatched = state; }
    void showMapPC(bool state) { _showMapPC = state; }
    void showMatchesPC(bool state) { _showMatchesPC = state; }
    void showLocalMapPC(bool state) { _showLocalMapPC = state; }
    void showKeyFrames(bool state) { _showKeyFrames = state; }
    void showCovisibilityGraph(bool state) { _showCovisibilityGraph = state; }
    void showSpanningTree(bool state) { _showSpanningTree = state; }
    void showLoopEdges(bool state) { _showLoopEdges = state; }
    void renderKfBackground(bool state) { _renderKfBackground = state; }
    void allowKfsAsActiveCam(bool state) { _allowKfsAsActiveCam = state; }

protected:
    //!flags, if keypoint positions should be rendered into current frame
    bool _showKeyPoints = false;
    //!flags, if keypoint positions of matches shoud be rendered into current frame
    bool _showKeyPointsMatched = true;
    //!flags, if all map points should be visualized
    bool _showMapPC = true;
    //!flags, if the subset of matched 3D points of the map points should be visualized
    bool _showMatchesPC = true;
    //!flags, if the local map points should be visualized
    bool _showLocalMapPC = false;
    //!flags, if keyframes should be shown
    bool _showKeyFrames = true;
    //!flags, if graphs should be shown
    bool _showCovisibilityGraph = false;
    bool _showSpanningTree = true;
    bool _showLoopEdges = true;
    //!flags, if keyframes backgound should be rendered
    bool _renderKfBackground = false;
    //!flags, if we allow to iterate through keyframes and set them as active scene cameras
    bool _allowKfsAsActiveCam = false;
};

#endif //SL_TRACKINGINFOSINTERFACE_H
