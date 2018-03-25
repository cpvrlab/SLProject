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

                for(int j=0; j < landmarks[i].size(); j++)
                {
                    // Landmark indexes from
                    // https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
                    if (j==36 || j==45 || j==48 || j==54 || j==30 || j==33 || j==8)
                         circle(imageRgb, landmarks[i][j], 3, cv::Scalar(0, 255, 0), -1);
                    else circle(imageRgb, landmarks[i][j], 3, cv::Scalar(0, 0, 255), -1);
                }
            }
        }
    }
    
    return false;
}
//-----------------------------------------------------------------------------
