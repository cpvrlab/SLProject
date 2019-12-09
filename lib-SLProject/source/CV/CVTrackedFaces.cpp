//#############################################################################
//  File:      CVTrackedFaces.cpp
//  Author:    Marcus Hudritsch
//  Date:      Spring 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/
#include <CVTrackedFaces.h>
#include <Utils.h>
#include <SLApplication.h>

//-----------------------------------------------------------------------------
//! Constructor for the facial landmark tracker
/*! The Constructor loads the training files for the face and the facial
landmarks. It also defines an averaged set of facial landmark points in 3D
used for pose estimation.
\param node Node to track and move (usually the camera for AR)
\param smoothlength length of averaging filter
\param faceClassifierFilename Name of the cascaded face training file
\param faceMarkModelFilename Name of the facial landmark training file
*/
CVTrackedFaces::CVTrackedFaces(int    smoothLenght,
                               string faceClassifierFilename,
                               string faceMarkModelFilename)
{
    // Load Haar cascade training file for the face detection
    if (!Utils::fileExists(faceClassifierFilename))
    {
        faceClassifierFilename = SLApplication::calibIniPath + faceClassifierFilename;
        if (!Utils::fileExists(faceClassifierFilename))
        {
            string msg = "CVTrackedFaces: File not found: " + faceClassifierFilename;
            Utils::exitMsg(msg.c_str(), __LINE__, __FILE__);
        }
    }
    _faceDetector = new CVCascadeClassifier(faceClassifierFilename);

    // Load facemark model file for the facial landmark detection
    if (!Utils::fileExists(faceMarkModelFilename))
    {
        faceMarkModelFilename = SLApplication::calibIniPath + faceMarkModelFilename;
        if (!Utils::fileExists(faceMarkModelFilename))
        {
            string msg = "CVTrackedFaces: File not found: " + faceMarkModelFilename;
            Utils::exitMsg(msg.c_str(), __LINE__, __FILE__);
        }
    }

    _facemark = cv::face::FacemarkLBF::create();
    _facemark->loadModel(faceMarkModelFilename);

    // Init averaged 2D facial landmark points
    _smoothLenght = smoothLenght;
    _avgPosePoints2D.emplace_back(AvgCVVec2f(smoothLenght, CVVec2f(0, 0))); // Nose tip
    _avgPosePoints2D.emplace_back(AvgCVVec2f(smoothLenght, CVVec2f(0, 0))); // Nose hole left
    _avgPosePoints2D.emplace_back(AvgCVVec2f(smoothLenght, CVVec2f(0, 0))); // Nose hole right
    _avgPosePoints2D.emplace_back(AvgCVVec2f(smoothLenght, CVVec2f(0, 0))); // Left eye left corner
    _avgPosePoints2D.emplace_back(AvgCVVec2f(smoothLenght, CVVec2f(0, 0))); // Left eye right corner
    _avgPosePoints2D.emplace_back(AvgCVVec2f(smoothLenght, CVVec2f(0, 0))); // Right eye left corner
    _avgPosePoints2D.emplace_back(AvgCVVec2f(smoothLenght, CVVec2f(0, 0))); // Right eye right corner
    _avgPosePoints2D.emplace_back(AvgCVVec2f(smoothLenght, CVVec2f(0, 0))); // Left mouth corner
    _avgPosePoints2D.emplace_back(AvgCVVec2f(smoothLenght, CVVec2f(0, 0))); // Right mouth corner

    _cvPosePoints2D.resize(_avgPosePoints2D.size(), CVPoint2f(0, 0));

    // Set 3D facial points in mm
    _cvPosePoints3D.push_back(CVPoint3f(.000f, .000f, .000f));    // Nose tip
    _cvPosePoints3D.push_back(CVPoint3f(-.015f, -.005f, -.018f)); // Nose hole left
    _cvPosePoints3D.push_back(CVPoint3f(.015f, -.005f, -.018f));  // Nose hole right
    _cvPosePoints3D.push_back(CVPoint3f(-.047f, .041f, -.036f));  // Left eye left corner
    _cvPosePoints3D.push_back(CVPoint3f(-.019f, .041f, -.033f));  // Left eye right corner
    _cvPosePoints3D.push_back(CVPoint3f(.019f, .041f, -.033f));   // Right eye left corner
    _cvPosePoints3D.push_back(CVPoint3f(.047f, .041f, -.036f));   // Right eye right corner
    _cvPosePoints3D.push_back(CVPoint3f(-.025f, -.035f, -.036f)); // Left Mouth corner
    _cvPosePoints3D.push_back(CVPoint3f(.025f, -.035f, -.036f));  // Right mouth corner
}
//-----------------------------------------------------------------------------
CVTrackedFaces::~CVTrackedFaces()
{
    delete _faceDetector;
}
//-----------------------------------------------------------------------------
//! Tracks the a face and its landmarks
/* The tracking is done by first detecting the face with a pretrained cascaded
face classifier implemented in OpenCV. The facial landmarks are detected with
the OpenCV face module using the facemarkLBF detector. More information about
OpenCV facial landmark detection can be found on:
https://www.learnopencv.com/facemark-facial-landmark-detection-using-opencv
\n
The pose estimation is done using cv::solvePnP with 9 facial landmarks in 3D 
and their corresponding 2D points detected by the cv::facemark detector. For
smoothing out the jittering we average the last few detections.
\param imageGray Image for processing
\param imageRgb Image for visualizations
\param calib Pointer to a valid camera calibration 
\param drawDetection Flag for drawing the detected obbjects
\param sv Pointer to the sceneview
*/
bool CVTrackedFaces::track(CVMat          imageGray,
                           CVMat          imageRgb,
                           CVCalibration* calib)
{
    assert(!imageGray.empty() && "ImageGray is empty");
    assert(!imageRgb.empty() && "ImageRGB is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");

    //////////////////
    // Detect Faces //
    //////////////////

    float startMS = _timer.elapsedTimeInMilliSec();

    // Detect faces
    CVVRect faces;
    int     min = (int)((float)imageGray.rows * 0.4f); // the bigger min the faster
    int     max = (int)((float)imageGray.rows * 0.8f); // the smaller max the faster
    CVSize  minSize(min, min);
    CVSize  maxSize(max, max);
    _faceDetector->detectMultiScale(imageGray, faces, 1.05, 3, 0, minSize, maxSize);

    // Enlarge the face rect at the bottom to cover also the chin
    for (auto& face : faces)
        face.height = (int)(face.height * 1.2f);

    float time2MS = _timer.elapsedTimeInMilliSec();
    CVTracked::detect1TimesMS.set(time2MS - startMS);

    //////////////////////
    // Detect Landmarks //
    //////////////////////

    CVVVPoint2f landmarks;
    bool        foundLandmarks = _facemark->fit(imageRgb, faces, landmarks);

    float time3MS = _timer.elapsedTimeInMilliSec();
    CVTracked::detect2TimesMS.set(time3MS - time2MS);
    CVTracked::detectTimesMS.set(time3MS - startMS);

    if (foundLandmarks)
    {
        for (unsigned long i = 0; i < landmarks.size(); i++)
        {
            // Landmark indexes from
            // https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
            _avgPosePoints2D[0].set(CVVec2f(landmarks[i][30].x, landmarks[i][30].y)); // Nose tip
            _avgPosePoints2D[1].set(CVVec2f(landmarks[i][31].x, landmarks[i][31].y)); // Nose hole left
            _avgPosePoints2D[2].set(CVVec2f(landmarks[i][35].x, landmarks[i][35].y)); // Nose hole right
            _avgPosePoints2D[3].set(CVVec2f(landmarks[i][36].x, landmarks[i][36].y)); // Left eye left corner
            _avgPosePoints2D[4].set(CVVec2f(landmarks[i][39].x, landmarks[i][39].y)); // Left eye right corner
            _avgPosePoints2D[5].set(CVVec2f(landmarks[i][42].x, landmarks[i][42].y)); // Right eye left corner
            _avgPosePoints2D[6].set(CVVec2f(landmarks[i][45].x, landmarks[i][45].y)); // Right eye right corner
            _avgPosePoints2D[7].set(CVVec2f(landmarks[i][48].x, landmarks[i][48].y)); // Left mouth corner
            _avgPosePoints2D[8].set(CVVec2f(landmarks[i][54].x, landmarks[i][54].y)); // Right mouth corner

            // Convert averaged 2D points to OpenCV points2d
            for (unsigned long p = 0; p < _avgPosePoints2D.size(); p++)
                _cvPosePoints2D[p] = CVPoint2f(_avgPosePoints2D[p].average()[0], _avgPosePoints2D[p].average()[1]);

            //delaunayTriangulate(imageRgb, landmarks[i], drawDetection);

            ///////////////////
            // Visualization //
            ///////////////////

            if (_drawDetection)
            {
                // Draw rectangle of detected face
                cv::rectangle(imageRgb, faces[i], cv::Scalar(255, 0, 0), 2);

                // Draw detected landmarks
                for (auto& j : landmarks[i])
                    cv::circle(imageRgb, j, 2, cv::Scalar(0, 0, 255), -1);

                // Draw averaged face points used for pose estimation
                for (unsigned long p = 0; p < _avgPosePoints2D.size(); p++)
                    cv::circle(imageRgb, _cvPosePoints2D[p], 5, cv::Scalar(0, 255, 0), 1);
            }

            // Do pose estimation for the first face found
            if (i == 0)
            {
                /////////////////////
                // Pose Estimation //
                /////////////////////

                startMS = _timer.elapsedTimeInMilliSec();

                //find the camera extrinsic parameters (rVec & tVec)
                CVMat rVec; // rotation angle vector as axis (length as angle)
                CVMat tVec; // translation vector
                bool  solved = solvePnP(CVMat(_cvPosePoints3D),
                                       CVMat(_cvPosePoints2D),
                                       calib->cameraMat(),
                                       calib->distortion(),
                                       rVec,
                                       tVec,
                                       false,
                                       cv::SOLVEPNP_EPNP);

                CVTracked::poseTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

                if (solved)
                {
                    _objectViewMat = createGLMatrix(tVec, rVec);
                    return true;
                }
            }
        }
    }

    return false;
}
//-----------------------------------------------------------------------------
// Returns the Delaunay triangulation on the points within the image
void CVTrackedFaces::delaunayTriangulate(CVMat             imageRgb,
                                         const CVVPoint2f& points,
                                         bool              drawDetection)
{
    // Get rect of image
    CVSize size = imageRgb.size();
    CVRect rect(0, 0, size.width, size.height);

    // Create an instance of Subdiv2D
    cv::Subdiv2D subdiv(rect);

    // Do Delaunay triangulation for the landmarks
    for (const CVPoint2f& point : points)
        if (rect.contains(point))
            subdiv.insert(point);

    // Add additional points in the corners and middle of the sides
    float w = (float)size.width;
    float h = (float)size.height;
    subdiv.insert(CVPoint2f(0.0f, 0.0f));
    subdiv.insert(CVPoint2f(w * 0.5f, 0.0f));
    subdiv.insert(CVPoint2f(w - 1.0f, 0.0f));
    subdiv.insert(CVPoint2f(w - 1.0f, h * 0.5f));
    subdiv.insert(CVPoint2f(w - 1.0f, h - 1.0f));
    subdiv.insert(CVPoint2f(w * 0.5f, h - 1.0f));
    subdiv.insert(CVPoint2f(0.0f, h - 1.0f));
    subdiv.insert(CVPoint2f(0.0f, h * 0.5f));

    // Draw Delaunay triangles
    if (drawDetection)
    {
        vector<cv::Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);
        CVVPoint pt(3);

        for (auto t : triangleList)
        {
            pt[0] = CVPoint(cvRound(t[0]), cvRound(t[1]));
            pt[1] = CVPoint(cvRound(t[2]), cvRound(t[3]));
            pt[2] = CVPoint(cvRound(t[4]), cvRound(t[5]));

            // Draw rectangles completely inside the image.
            if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
            {
                line(imageRgb, pt[0], pt[1], cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
                line(imageRgb, pt[1], pt[2], cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
                line(imageRgb, pt[2], pt[0], cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
            }
        }
    }
}
//-----------------------------------------------------------------------------
