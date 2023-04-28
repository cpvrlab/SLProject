//#############################################################################
//  File:      cv13_FaceTracking.cpp
//  Purpose:   Minimal OpenCV application for face Tracking in video
//  Source:    https://docs.opencv.org/3.0-beta/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
//  Date:      Authumn 2017
//#############################################################################

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//----------------------------------------------------------------------------
// Globals
// Note for Visual Studio: You must set the Working Directory to $(TargetDir)
// with: Right Click on Project > Properties > Debugging
static String projectRoot       = String(SL_PROJECT_ROOT);
static String face_cascade_name = projectRoot + "/data/opencv/haarcascades/haarcascade_frontalface_alt.xml";
static String eyes_cascade_name = projectRoot + "/data/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

static CascadeClassifier face_cascade;
static CascadeClassifier eyes_cascade;
static String            window_name = "Capture - Face detection";

//-----------------------------------------------------------------------------
void detectFaceAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat               frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    face_cascade.detectMultiScale(frame_gray,
                                  faces,
                                  1.1,
                                  2,
                                  0 | CASCADE_SCALE_IMAGE,
                                  Size(30, 30));

    for (size_t i = 0; i < faces.size(); i++)
    {
        rectangle(frame, faces[i], Scalar(255, 0, 255), 2);

        Mat               faceROI = frame_gray(faces[i]);
        std::vector<Rect> eyes;

        // In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI,
                                      eyes,
                                      1.1,
                                      2,
                                      0 | CASCADE_SCALE_IMAGE,
                                      Size(30, 30));

        for (size_t j = 0; j < eyes.size(); j++)
        {
            eyes[j].x += faces[i].x;
            eyes[j].y += faces[i].y;
            rectangle(frame, eyes[j], Scalar(255, 0, 0), 2);
        }
    }

    imshow(window_name, frame);
}

//-----------------------------------------------------------------------------
int main()
{
    // Be aware that on Windows not more than one process can access the camera at the time.
    // Be aware that on many OS you have to grant access rights to the camera system
    VideoCapture capture;
    Mat          frame;

    // 1. Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        printf("Error loading face cascade\n");
        return -1;
    };
    if (!eyes_cascade.load(eyes_cascade_name))
    {
        printf("Error loading eyes cascade\n");
        return -1;
    };

    // 2. Read the video stream
    capture.open(0);
    if (!capture.isOpened())
    {
        printf("Error opening video capture\n");
        return -1;
    }

    while (capture.read(frame))
    {
        if (frame.empty())
        {
            printf("No captured frame -- Break!");
            break;
        }

        detectFaceAndDisplay(frame);

        // Wait for key to exit loop
        if (waitKey(10) != -1)
            return 0;
    }
    return 0;
}

//-----------------------------------------------------------------------------
