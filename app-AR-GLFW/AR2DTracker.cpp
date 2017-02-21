//#############################################################################
//  File:      AR2DTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "AR2DTracker.h"
#include <SLMaterial.h>
#include <SLAssimpImporter.h>
#include <SLBox.h>
#include <ARSceneView.h>

#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/video.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define AR_SAVE_DEBUG_IMAGES 0
#define AR_KNN_MATCH 1
#define AR_USE_HOMOGRAPHY 0

//-----------------------------------------------------------------------------
AR2DTracker::AR2DTracker() :
    _posInitialized(false),
    _posValid(false),
    _node(nullptr)
{
}
//-----------------------------------------------------------------------------
bool AR2DTracker::init()
{
    //load ARMap
    _map.loadFromFile("map2d");

    //initialize feature detector depending on _map type
    if(_map.type == AR2DMap::AR_SURF)
    {
        _detector = SURF::create(_map.minHessian);
        //initialize matcher
        _matcher = DescriptorMatcher::create("BruteForce");
    }
    else if(_map.type == AR2DMap::AR_ORB)
    {
        /*The maximum number of features to retain.*/
        int nFeatures = 1000;
        /*Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
        pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
        will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
        will mean that to cover certain scale range you will need more pyramid levels and so the speed
        will suffer.*/
        float scaleFactor = 1.2f;
        /*The number of pyramid levels. The smallest level will have linear size equal to
        input_image_linear_size/pow(scaleFactor, nlevels).*/
        int nlevels = 8;
        /*This is size of the border where the features are not detected. It should
        roughly match the patchSize parameter.*/
        int edgeThreshold = 31;
        //It should be 0 in the current implementation.
        int firstLevel = 0;
        /*The number of points that produce each element of the oriented BRIEF descriptor. The
        default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
        so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
        random points (of course, those point coordinates are random, but they are generated from the
        pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
        rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
        output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
        denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
        bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).*/
        int WTA_K = 2;
        /*The default HARRIS_SCORE means that Harris algorithm is used to rank features
        (the score is written to KeyPoint::score and is used to retain best nfeatures features);
        FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
        but it is a little faster to compute.*/
        int scoreType = ORB::HARRIS_SCORE;
        /*size of the patch used by the oriented BRIEF descriptor. Of course, on smaller
        pyramid layers the perceived image area covered by a feature will be larger.*/
        int patchSize = 31;
        int fastThreshold = 20;
        _detector = ORB::create(nFeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
        //initialize matcher
        _matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }
    return true; //???
}
//-----------------------------------------------------------------------------
bool AR2DTracker::track(cv::Mat image, 
                        SLCVCalibration& calib)
{
    //reset flag
    _posValid = false;

    Mat gray, rVec, rMat, tVec;
    cvtColor(image, gray, COLOR_RGB2GRAY);

    //detect features in video stream
    _detector->detectAndCompute(gray, Mat(), _sceneKeypoints, _sceneDescriptors);

    #if AR_SAVE_DEBUG_IMAGES
    Mat sceneKeyPtsImg;
    cv::drawKeypoints(_image, _sceneKeypoints, sceneKeyPtsImg);
    cv::imwrite("sceneKeyPtsImg.bmp", sceneKeyPtsImg);

    Mat mapKeyPtsImg;
    cv::drawKeypoints(_map.image, _map.keypoints, mapKeyPtsImg);
    cv::imwrite("mapKeyPtsImg.bmp", mapKeyPtsImg);
    #endif
    _mapPts.clear();
    _scenePts.clear();
    std::vector< DMatch > goodMatches;

    //if we have no initial position
    if(!_posInitialized)
    {
        #if AR_KNN_MATCH
        std::vector<vector<DMatch> > matches;
        //find two nearest neighbors
        _matcher->knnMatch(_sceneDescriptors, _map.descriptors, matches, 2);
        //apply sift ratio test
        const float ratio = 0.8f; // As in Lowe's paper
        for (int i = 0; i < matches.size(); ++i)
        {
            //we want matches, which have no near neighbors and are kind of unique
            if (matches[i][0].distance < ratio * matches[i][1].distance)
            {
                goodMatches.push_back(matches[i][0]);
            }
        }
        #else
        //match features
        vector<DMatch> matches;
        _matcher->match(_sceneDescriptors, _map.descriptors, matches);

        //filter matches depending on distance
        double maxDist = 0; double minDist = 100;
        for(int i = 0; i < _map.descriptors.rows; i++)
        { double dist = matches[i].distance;
          if(dist < minDist) minDist = dist;
          if(dist > maxDist) maxDist = dist;
        }
        printf("-- Max dist : %f \n", maxDist);
        printf("-- Min dist : %f \n", minDist);
        //good matches (distance is less than 3*minDist)

        for(int i = 0; i < _map.descriptors.rows; i++)
        {
            if(matches[i].distance < 3 * minDist)
            {
                goodMatches.push_back(matches[i]);
            }
        }
        #endif

        //extract 2d pts depending on good matches
        for(size_t i = 0; i < goodMatches.size(); i++)
        {
          //-- Get the keypoints from the good matches
          const Point2f& pt = _map.pts[ goodMatches[i].trainIdx ];
          _mapPts.push_back(Point3f(pt.x, pt.y, 0.0f));
          const Point2f& scPt = _sceneKeypoints[ goodMatches[i].queryIdx ].pt;
          _scenePts.push_back(scPt);
        }

        #if AR_SAVE_DEBUG_IMAGES
        Mat imgMatches;
        drawMatches(_image, _sceneKeypoints, _map.image, _map.keypoints,
                     goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1),
                     std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        imwrite("Good_matches.png", imgMatches);
        #endif
    }
    //else if we have an initial position
    else
    {
        //use optical flow for feature tracking


    }

    if(_scenePts.size() > 10)
    {
        std::vector< DMatch > realGoodMatches;

        #if AR_USE_HOMOGRAPHY
        //estimate camera position with homography
        Mat H = findHomography(_mapPts, _scenePts, RANSAC);
        //cout << "H: " << H << endl;
        std::vector<cv::Mat> Rs, Ts, Ns;
        cv::decomposeHomographyMat(H, _intrinsics, Rs, Ts, Ns);
        for(size_t i=0; i < Rs.size(); ++i)
        {
            cout << "Homography Rs:" << Rs[i] << endl;
        }

        //        cv::Mat homRVect;
        //        cv::Rodrigues(Rs[2], homRVect);
        //        cout << "Homography homRVect:" << homRVect << endl;

        //cout << "Homography Rs:" << Rs[i] << endl;
        cout << "" << endl;
        #else
        //Mat inliers_idx;
        std::vector<int> inliers;
        //solvePnP(_mapPts, _scenePts, _intrinsics, _distortion, _rVec, _tVec, false, SOLVEPNP_ITERATIVE);
        solvePnPRansac(_mapPts, 
                       _scenePts, 
                       calib.cameraMat(), 
                       calib.distortion(), 
                       rVec, 
                       tVec, 
                       true, 
                       1000, 
                       1.0, 
                       0.99, 
                       inliers,
                        /*cv::SOLVEPNP_P3P*/ /*cv::SOLVEPNP_EPNP*/  SOLVEPNP_ITERATIVE /*cv::SOLVEPNP_DLS*/ /*cv::SOLVEPNP_UPNP*/);

        for(size_t i=0; i < inliers.size(); ++i)
        {
            unsigned int idx = inliers[i];
            realGoodMatches.push_back(goodMatches[idx]);
        }
        #endif


        #if AR_SAVE_DEBUG_IMAGES
        Mat imgMatches;
        drawMatches(_image, _sceneKeypoints, _map.image, _map.keypoints,
                     realGoodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1),
                     std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imwrite("Real_Good_matches.png", imgMatches);
        #endif

        //check if we have enough inliers
        //if(countNonZero(Mat(matchMask)) < 15)
        if(realGoodMatches.size() < 10)
        {
            _posValid = false;
        }
        else
        {
            _posValid = true;

            // Convert cv translation & rotation to OpenGL transform matrix
            SLMat4f ovm = createGLMatrix(tVec, rVec);
        }
    }
    return true; //???
}
//-----------------------------------------------------------------------------
void AR2DTracker::updateSceneView(ARSceneView* sv)
{
    if(!_node)
    {
        //create new box
        SLMaterial* rMat = new SLMaterial("rMat", SLCol4f(1.0f,0.7f,0.7f));
        _node = new SLNode("Box");

        // load coordinate axis arrows
        SLAssimpImporter importer;
        SLNode* axesNode = importer.load("FBX/Axes/axes_blender.fbx");
        axesNode->scale(0.3f);
        _node->addChild(axesNode);

        float edgeLength = 0.16f / 2;
        _node->addMesh(new SLBox(0.0f, 0.0f, 0.0f, edgeLength, edgeLength, edgeLength, "Box", rMat));

        SLScene::current->root3D()->addChild(_node);
        _node->updateAABBRec();
    }

    if(_posValid)
    {
        //invert view matrix because we want to set the camera object matrix
        SLMat4f camOm = _viewMat.inverse();
        //update camera with calculated view matrix:
        sv->camera()->om(camOm);
        //set node visible
        _node->setDrawBitsRec(SL_DB_HIDDEN, false);
    }
    else
        //set invisible
        _node->setDrawBitsRec(SL_DB_HIDDEN, true);
}
//-----------------------------------------------------------------------------
void AR2DTracker::unloadSGObjects()
{
    if(_node)
    {
        SLNode* parent = _node->parent();
        parent->deleteChild(_node);
        _node = nullptr;
        parent->updateAABBRec();
    }
}

