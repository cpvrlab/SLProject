//#############################################################################
//  File:      SLCVTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVCHESSBOARDTRACKER_H
#define SLCVCHESSBOARDTRACKER_H

#include <SLCVTracker.h>

//-----------------------------------------------------------------------------
/*!
Chessboard tracking class
*/
class SLCVTrackerChessboard : public SLCVTracker
{
    public:
                SLCVTrackerChessboard   (SLNode* node);
               ~SLCVTrackerChessboard   () {;}
        bool    track                   (cv::Mat image, 
                                         SLCVCalibration& calib,
                                         SLVSceneView& sceneViews);
    private:
        SLfloat              _edgeLengthM;  //<! Length of chessboard square in meters
        vector<cv::Point3d>  _boardPoints;  //<! chessboard corners in world coordinate system
        cv::Size             _boardSize;    //<! NO. of inner chessboard corners
};
//-----------------------------------------------------------------------------

#endif // SLCVCHESSBOARDTRACKER_H
