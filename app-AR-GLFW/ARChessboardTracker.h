//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef ARCHESSBOARDTRACKER_H
#define ARCHESSBOARDTRACKER_H

#include <ARTracker.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------

/*!
Parameter class for chessboard tracking parameter
*/
class ARChessboardParams
{
public:
    ARChessboardParams() :
        boardWidth(6),
        boardHeight(8),
        edgeLengthM(0.035f),
        filename("chessboard_detector_params.yml")
    {}

    bool loadFromFile(string paramsDir)
    {
        cv::FileStorage fs( paramsDir + filename, cv::FileStorage::READ);
        if(!fs.isOpened())
        {
            cout << "Could not find parameter file for Chessboard tracking!" << endl;
            cout << "Tried " << paramsDir + filename << endl;
            return false;
        }
        fs["boardWidth"] >> boardWidth;
        fs["boardHeight"] >> boardHeight;
        fs["edgeLengthM"] >> edgeLengthM;
        return true;
    }

    //number of inner chessboard corners in width direction
    int boardWidth;
    //number of inner chessboard corners in height direction
    int boardHeight;
    //edge length of chessboard square in meters
    float edgeLengthM;
    //parameter file name
    string filename;
};

//-----------------------------------------------------------------------------

/*!
Chessboard tracking class
*/
class ARChessboardTracker : public ARTracker
{
public:
    ARChessboardTracker(cv::Mat intrinsics, cv::Mat distoriton);

    bool init(string paramsFileDir) override;
    bool track() override;
    void updateSceneView( ARSceneView* sv ) override;
    void unloadSGObjects() override;

    const ARChessboardParams& params() const { return _p; }

private:
    //chessboard corners in world coordinate system
    vector<cv::Point3d> _boardPoints;
    //calculated image points in findChessboardCorners
    vector<cv::Point2d> _imagePoints;
    //Parameter class instance
    ARChessboardParams _p;

    SLNode* _node;
    bool _cbVisible;
};

#endif // ARCHESSBOARDTRACKER_H
