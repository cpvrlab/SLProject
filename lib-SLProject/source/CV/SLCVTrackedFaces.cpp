//#############################################################################
//  File:      SLCVTrackedFaces.cpp
//  Author:    Marcus Hudritsch
//  Date:      Spring 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/
#include <SLApplication.h>
#include <SLSceneView.h>
#include <SLCVTrackedFaces.h>

//-----------------------------------------------------------------------------
SLCVTrackedFaces::SLCVTrackedFaces(SLNode* node) :
                 SLCVTracked(node)
{
    // Load Haar cascade training file
    SLstring filename = "haarcascade_frontalface_alt2.xml";
    if (!SLFileSystem::fileExists(filename))
    {   filename = SLCVCalibration::calibIniPath + filename;
        if (!SLFileSystem::fileExists(filename))
        {   SLstring msg = "SLCVTrackedFaces: File not found: " + filename;
            SL_EXIT_MSG(msg.c_str());
        }
    }
    _faceDetector = new SLCVCascadeClassifier(filename);


    // Load facemark model file
    filename = "lbfmodel.yaml";
    if (!SLFileSystem::fileExists(filename))
    {   filename = SLCVCalibration::calibIniPath + filename;
        if (!SLFileSystem::fileExists(filename))
        {   SLstring msg = "SLCVTrackedFaces: File not found: " + filename;
            SL_EXIT_MSG(msg.c_str());
        }
    }

    _facemark = cv::face::FacemarkLBF::create();
    _facemark->loadModel(filename);

    // Init averaged 2D points
    _points2D.push_back(SLAvgVec2f(5, SLVec2f(0,0))); // Nose tip
    _points2D.push_back(SLAvgVec2f(5, SLVec2f(0,0))); // Chin
    _points2D.push_back(SLAvgVec2f(5, SLVec2f(0,0))); // Left eye left corner
    _points2D.push_back(SLAvgVec2f(5, SLVec2f(0,0))); // Right eye right corner
    _points2D.push_back(SLAvgVec2f(5, SLVec2f(0,0))); // Left mouth corner
    _points2D.push_back(SLAvgVec2f(5, SLVec2f(0,0))); // Right mouth corner
}
//-----------------------------------------------------------------------------
SLCVTrackedFaces::~SLCVTrackedFaces()
{
    delete _faceDetector;
}
//-----------------------------------------------------------------------------
//! Tracks the ...
/* The tracking ...
*/
SLbool SLCVTrackedFaces::track(SLCVMat imageGray,
                              SLCVMat imageRgb,
                              SLCVCalibration* calib,
                              SLbool drawDetection,
                              SLSceneView* sv)
{
    assert(!imageGray.empty() && "ImageGray is empty");
    assert(!imageRgb.empty() && "ImageRGB is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");
   
    //////////////////
    // Detect Faces //
    //////////////////
    
    SLScene* s = SLApplication::scene;
    SLfloat startMS = s->timeMilliSec();
    
    // Detect faces
    SLCVVRect faces;
    SLint min = (SLint)(imageGray.rows*0.4f); // the bigger min the faster
    SLint max = (SLint)(imageGray.rows*0.8f); // the smaller max the faster
    SLCVSize minSize(min, min);
    SLCVSize maxSize(max, max);
    _faceDetector->detectMultiScale(imageGray, faces, 1.1, 3, 0, minSize, maxSize);

    SLfloat time2MS = s->timeMilliSec();
    s->detect1TimesMS().set(time2MS-startMS);
    
    //////////////////////
    // Detect Landmarks //
    //////////////////////

    SLCVVVPoint2f landmarks;
    SLbool foundLandmarks = _facemark->fit(imageRgb, faces, landmarks);

    SLfloat time3MS = s->timeMilliSec();
    s->detect2TimesMS().set(time3MS-time2MS);
    s->detectTimesMS().set(time3MS-startMS);
    
    if(foundLandmarks)
    {
        if (drawDetection)
        {
            for(int i = 0; i < landmarks.size(); i++)
            {
                rectangle(imageRgb, faces[i], cv::Scalar(255, 0, 255), 2);

                // Landmark indexes from
                // https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
                _points2D[0].set(SLVec2f(landmarks[i][30].x, landmarks[i][30].y)); // Nose tip
                _points2D[1].set(SLVec2f(landmarks[i][ 8].x, landmarks[i][ 8].y)); // Chin
                _points2D[2].set(SLVec2f(landmarks[i][36].x, landmarks[i][36].y)); // Left eye left corner
                _points2D[3].set(SLVec2f(landmarks[i][45].x, landmarks[i][45].y)); // Right eye right corner
                _points2D[4].set(SLVec2f(landmarks[i][48].x, landmarks[i][48].y)); // Left mouth corner
                _points2D[5].set(SLVec2f(landmarks[i][54].x, landmarks[i][54].y)); // Right mouth corner

                for(int j=0; j < landmarks[i].size(); j++)
                    circle(imageRgb, landmarks[i][j], 2, cv::Scalar(0, 0, 255), -1);

                for(int p=0; p < _points2D.size(); p++)
                {
                    SLCVPoint2f cvP2D(_points2D[i].average().x, _points2D[i].average().y);
                    cv::circle(imageRgb, cvP2D, 5, cv::Scalar(0, 255, 0), -1);
                }
            }
        }
    }
    
    return false;
}
//-----------------------------------------------------------------------------
