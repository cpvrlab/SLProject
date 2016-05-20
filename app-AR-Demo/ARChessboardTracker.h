//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef ARCHESSBOARDTRACKER_H
#define ARCHESSBOARDTRACKER_H

#include <ARTracker.h>
#include <SLNode.h>

//Parameter class for chessboard tracking parameter
class ARChessboardParams
{
public:
    ARChessboardParams() :
        boardWidth(6),
        boardHeight(8),
        edgeLengthM(0.035f)
    {}

    bool loadFromFile(string filename)
    {
        //if not loading
        {
            cout << "Could not find parameter file for Chessboard tracking!" << endl;
            cout << "Tried ..." << endl;
        }

        return true;
    }

    int boardWidth;
    int boardHeight;
    //chessboard size (number of inner squares)
    //cv::Size cbSize;
    //edge length of chessboard square in meters
    float edgeLengthM;
    //parameter file name
    string filename;
};

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
    //void    initChessboardTracking  (string camParamsFilename, int boardHeight, int boardWidth, float   edgeLengthM );
    //bool    trackChessboard     ();

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
