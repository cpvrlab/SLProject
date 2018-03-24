//#############################################################################
//  File:      SLCVTrackedFace.cpp
//  Author:    Marcus Hudritsch
//  Date:      Spring 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/
#include <SLApplication.h>
#include <SLSceneView.h>
#include <SLCVTrackedFace.h>

//-----------------------------------------------------------------------------
SLCVTrackedFace::SLCVTrackedFace(SLNode* node) :
                 SLCVTracked(node)
{
    _faceDetector = new SLCVCascadeClassifier("../_data/opencv/haarcascades/haarcascade_frontalface_alt2.xml");
    _facemark = cv::face::FacemarkLBF::create();
    _facemark->loadModel("../_data/calibrations/lbfmodel.yaml");
}
//-----------------------------------------------------------------------------
SLCVTrackedFace::~SLCVTrackedFace()
{
    delete _faceDetector;
    delete _facemark;
}
//-----------------------------------------------------------------------------
//! Tracks the ...
/* The tracking ...
*/
SLbool SLCVTrackedFace::track(SLCVMat imageGray,
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
   
    ////////////
    // Detect //
    ////////////
    
    SLScene* s = SLApplication::scene;
    SLfloat startMS = s->timeMilliSec();
    
    // Detect faces
    SLCVVRect faces;
    _faceDetector->detectMultiScale(imageGray, faces);
    
    // Detect landmarks in multiple faces
    SLCVVVPoint2f landmarks;
    SLbool foundLandmarks = _facemark->fit(imageRgb, faces, landmarks);
    
    s->detectTimesMS().set(s->timeMilliSec()-startMS);
    
    if(foundLandmarks)
    {
        if (drawDetection)
        {   for(int i = 0; i < landmarks.size(); i++)
                for(int j=0; j < landmarks[i].size(); j++)
                    circle(imageRgb, landmarks[i][j], 3, cv::Scalar(0,0,255), -1);
        }
    }
    
    return false;
}
//-----------------------------------------------------------------------------
