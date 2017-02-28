//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Goettlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef ARCHESSBOARDTRACKER_H
#define ARCHESSBOARDTRACKER_H

#include <ARTracker.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
/*!
Chessboard tracking class
*/
class ARChessboardTracker : public ARTracker
{
    public:
                    ARChessboardTracker (){;}
        bool        init                () override;
        bool        track               (cv::Mat image, 
                                         SLCVCalibration* calib) override;
        void        updateSceneView     (ARSceneView* sv) override;
        void        unloadSGObjects     () override;

    private:
        SLCVVPoint3f    _boardPoints3D; //<! chessboard corners in world coordinate system
        SLCVSize        _boardSize;     //<! NO. of inner chessboard corners
        SLfloat         _edgeLengthM;   //<! Length of chessboard square
};
//-----------------------------------------------------------------------------

#endif // ARCHESSBOARDTRACKER_H
