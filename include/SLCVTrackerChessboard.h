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

#include <SLCV.h>
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
                                         SLSceneView* sv);
    private:
        SLfloat         _edgeLengthM;   //<! Length of chessboard square in meters
        SLCVVPoint3d    _boardPoints;   //<! chessboard corners in world coordinate system
        SLCVSize        _boardSize;     //<! NO. of inner chessboard corners
};
//-----------------------------------------------------------------------------

#endif // SLCVCHESSBOARDTRACKER_H
