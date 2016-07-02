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
class SLCVChessboardTracker : public SLCVTracker
{
    public:
                SLCVChessboardTracker   (SLNode* node) : SLCVTracker(node){;}
        bool    init                    (string paramsFileDir) override;
        bool    track                   (cv::Mat image, 
                                         SLCVCalibration& calib,
                                         SLSceneView* sv) = 0;

    private:
        vector<cv::Point3d> _boardPoints;   //<! chessboard corners in world coordinate system
        cv::Size            _boardSize;     //<! NO. of inner chessboard corners
        SLfloat             _edgeLengthM;   //<! Length of chessboard square
};
//-----------------------------------------------------------------------------

#endif // SLCVCHESSBOARDTRACKER_H
