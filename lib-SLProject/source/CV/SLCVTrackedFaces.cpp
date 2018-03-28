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
SLCVTrackedFaces::SLCVTrackedFaces(SLNode*  node,
                                   SLint    smoothLenght,
                                   SLstring faceClassifierFilename,
                                   SLstring faceMarkModelFilenema) :
                  SLCVTracked(node)
{
    // Load Haar cascade training file for the face detection
    if (!SLFileSystem::fileExists(faceClassifierFilename))
    {   faceClassifierFilename = SLCVCalibration::calibIniPath + faceClassifierFilename;
        if (!SLFileSystem::fileExists(faceClassifierFilename))
        {   SLstring msg = "SLCVTrackedFaces: File not found: " + faceClassifierFilename;
            SL_EXIT_MSG(msg.c_str());
        }
    }
    _faceDetector = new SLCVCascadeClassifier(faceClassifierFilename);

    // Load facemark model file for the facial landmark detection
    if (!SLFileSystem::fileExists(faceMarkModelFilenema))
    {   faceMarkModelFilenema = SLCVCalibration::calibIniPath + faceMarkModelFilenema;
        if (!SLFileSystem::fileExists(faceMarkModelFilenema))
        {   SLstring msg = "SLCVTrackedFaces: File not found: " + faceMarkModelFilenema;
            SL_EXIT_MSG(msg.c_str());
        }
    }

    _facemark = cv::face::FacemarkLBF::create();
    _facemark->loadModel(faceMarkModelFilenema);

    // Init averaged 2D facial landmark points
    _smoothLenght = smoothLenght;
    _avgPosePoints2D.push_back(SLAvgVec2f(smoothLenght, SLVec2f::ZERO)); // Nose tip
    _avgPosePoints2D.push_back(SLAvgVec2f(smoothLenght, SLVec2f::ZERO)); // Nose hole left
    _avgPosePoints2D.push_back(SLAvgVec2f(smoothLenght, SLVec2f::ZERO)); // Nose hole right
    _avgPosePoints2D.push_back(SLAvgVec2f(smoothLenght, SLVec2f::ZERO)); // Left eye left corner
    _avgPosePoints2D.push_back(SLAvgVec2f(smoothLenght, SLVec2f::ZERO)); // Left eye right corner
    _avgPosePoints2D.push_back(SLAvgVec2f(smoothLenght, SLVec2f::ZERO)); // Right eye left corner
    _avgPosePoints2D.push_back(SLAvgVec2f(smoothLenght, SLVec2f::ZERO)); // Right eye right corner
    _avgPosePoints2D.push_back(SLAvgVec2f(smoothLenght, SLVec2f::ZERO)); // Left mouth corner
    _avgPosePoints2D.push_back(SLAvgVec2f(smoothLenght, SLVec2f::ZERO)); // Right mouth corner
    
    _cvPosePoints2D.resize(_avgPosePoints2D.size(), SLCVPoint2f(0,0));

    // Set 3D facial points in mm
    _cvPosePoints3D.push_back(SLCVPoint3f( .000f,  .000f,  .000f)); // Nose tip
    _cvPosePoints3D.push_back(SLCVPoint3f(-.015f, -.005f, -.018f)); // Nose hole left
    _cvPosePoints3D.push_back(SLCVPoint3f( .015f, -.005f, -.018f)); // Nose hole right
    _cvPosePoints3D.push_back(SLCVPoint3f(-.047f,  .041f, -.036f)); // Left eye left corner
    _cvPosePoints3D.push_back(SLCVPoint3f(-.019f,  .041f, -.033f)); // Left eye right corner
    _cvPosePoints3D.push_back(SLCVPoint3f( .019f,  .041f, -.033f)); // Right eye left corner
    _cvPosePoints3D.push_back(SLCVPoint3f( .047f,  .041f, -.036f)); // Right eye right corner
    _cvPosePoints3D.push_back(SLCVPoint3f(-.025f, -.035f, -.036f)); // Left Mouth corner
    _cvPosePoints3D.push_back(SLCVPoint3f( .025f, -.035f, -.036f)); // Right mouth corner
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
    _faceDetector->detectMultiScale(imageGray, faces, 1.05, 3, 0, minSize, maxSize);

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
        for(int i = 0; i < landmarks.size(); i++)
        {
            // Landmark indexes from
            // https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
            _avgPosePoints2D[0].set(SLVec2f(landmarks[i][30].x, landmarks[i][30].y)); // Nose tip
            _avgPosePoints2D[1].set(SLVec2f(landmarks[i][31].x, landmarks[i][31].y)); // Nose hole left
            _avgPosePoints2D[2].set(SLVec2f(landmarks[i][35].x, landmarks[i][35].y)); // Nose hole right
            _avgPosePoints2D[3].set(SLVec2f(landmarks[i][36].x, landmarks[i][36].y)); // Left eye left corner
            _avgPosePoints2D[4].set(SLVec2f(landmarks[i][39].x, landmarks[i][39].y)); // Left eye right corner
            _avgPosePoints2D[5].set(SLVec2f(landmarks[i][42].x, landmarks[i][42].y)); // Right eye left corner
            _avgPosePoints2D[6].set(SLVec2f(landmarks[i][45].x, landmarks[i][45].y)); // Right eye right corner
            _avgPosePoints2D[7].set(SLVec2f(landmarks[i][48].x, landmarks[i][48].y)); // Left mouth corner
            _avgPosePoints2D[8].set(SLVec2f(landmarks[i][54].x, landmarks[i][54].y)); // Right mouth corner

            // Converte averaged 2D points to OpenCV points2d
            for(SLint p=0; p < _avgPosePoints2D.size(); p++)
                _cvPosePoints2D[p] = SLCVPoint2f(_avgPosePoints2D[p].average().x, _avgPosePoints2D[p].average().y);
            
            //delaunayTriangulate(imageRgb, landmarks[i], drawDetection);
            
            ///////////////////
            // Visualization //
            ///////////////////
            
            if (drawDetection)
            {
                // Draw rectangle of detected face
                rectangle(imageRgb, faces[i], cv::Scalar(255, 0, 0), 2);
                
                // Draw detected landmarks
                for(int j=0; j < landmarks[i].size(); j++)
                    cv::circle(imageRgb, landmarks[i][j], 2, cv::Scalar(0, 0, 255), -1);
                
                // Draw averaged face points used for pose estimation
                for(int p=0; p < _avgPosePoints2D.size(); p++)
                    cv::circle(imageRgb, _cvPosePoints2D[p], 5, cv::Scalar(0, 255, 0), 1);
            }
            
            // Do pose estimation for the first face found
            if (i==0)
            {
                /////////////////////
                // Pose Estimation //
                /////////////////////
                
                startMS = s->timeMilliSec();
                
                //find the camera extrinsic parameters (rVec & tVec)
                SLCVMat rVec; // rotation angle vector as axis (length as angle)
                SLCVMat tVec; // translation vector
                SLbool solved = solvePnP(SLCVMat(_cvPosePoints3D),
                                         SLCVMat(_cvPosePoints2D),
                                         calib->cameraMat(),
                                         calib->distortion(),
                                         rVec,
                                         tVec,
                                         false,
                                         cv::SOLVEPNP_EPNP);
                
                s->poseTimesMS().set(s->timeMilliSec() - startMS);
                
                if (solved)
                {
                    _objectViewMat = createGLMatrix(tVec, rVec);
                    
                    // set the object matrix depending if the
                    // tracked node is attached to a camera or not
                    if (typeid(*_node)==typeid(SLCamera))
                        _node->om(_objectViewMat.inverted());
                    else
                    {
                        _node->om(calcObjectMatrix(sv->camera()->om(), _objectViewMat));
                        _node->setDrawBitsRec(SL_DB_HIDDEN, false);
                    }
                    return true;
                }
            }
        }
    }
    
    return false;
}
//-----------------------------------------------------------------------------
// Returns the Delaunay triangulation on the points within the image
void SLCVTrackedFaces::delaunayTriangulate(SLCVMat imageRgb,
                                           SLCVVPoint2f points,
                                           SLbool drawDetection)
{
    // Get rect of image
    SLCVSize size = imageRgb.size();
    SLCVRect rect(0, 0, size.width, size.height);
 
    // Create an instance of Subdiv2D
    cv::Subdiv2D subdiv(rect);
    
    // Do Delaunay trianglulation for the landmarks
    for(SLCVPoint2f point : points)
        if (rect.contains(point))
            subdiv.insert(point);
    
    // Add additional points in the corners and middle of the sides
    subdiv.insert(SLCVPoint2f(0,0));
    subdiv.insert(SLCVPoint2f(size.width/2,0));
    subdiv.insert(SLCVPoint2f(size.width-1,0));
    subdiv.insert(SLCVPoint2f(size.width-1,size.height/2));
    subdiv.insert(SLCVPoint2f(size.width-1,size.height-1));
    subdiv.insert(SLCVPoint2f(size.width/2,size.height-1));
    subdiv.insert(SLCVPoint2f(0,size.height-1));
    subdiv.insert(SLCVPoint2f(0,size.height/2));

    // Draw Delaunay triangles
    if (drawDetection)
    {
        vector<cv::Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);
        SLCVVPoint pt(3);
     
        for( size_t i = 0; i < triangleList.size(); i++ )
        {
            cv::Vec6f t = triangleList[i];
            pt[0] = SLCVPoint(cvRound(t[0]), cvRound(t[1]));
            pt[1] = SLCVPoint(cvRound(t[2]), cvRound(t[3]));
            pt[2] = SLCVPoint(cvRound(t[4]), cvRound(t[5]));
            
            // Draw rectangles completely inside the image.
            if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
            {
                line(imageRgb, pt[0], pt[1], cv::Scalar(255, 255, 255), 1, CV_AA, 0);
                line(imageRgb, pt[1], pt[2], cv::Scalar(255, 255, 255), 1, CV_AA, 0);
                line(imageRgb, pt[2], pt[0], cv::Scalar(255, 255, 255), 1, CV_AA, 0);
            }
        }
    }
}
//-----------------------------------------------------------------------------
