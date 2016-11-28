//#############################################################################
//  File:      SLCVFeatureMapper2D.cpp
//  Author:    Michael Göttlicher, Marcus Hudritsch
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

#include <SLCVFeatureMapper2D.h>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

//-----------------------------------------------------------------------------
SLCVFeatureMapper2D::SLCVFeatureMapper2D() :
    _state(IDLE)
{
}
//-----------------------------------------------------------------------------
void SLCVFeatureMapper2D::create(SLCVMat imageGray, 
                                 SLfloat offsetXMM, 
                                 SLfloat offsetYMM,
                                 SLstring filename,
                                 SLCVFeatureType type)
{
    //instantiate feature detector depending on type
    cv::Ptr<FeatureDetector> detector;
    switch(type)
    {
        case FT_ORB:
        {
            // The maximum number of features to retain.
            SLint nFeatures = 1000;
            
            /* Pyramid decimation ratio, greater than 1. scaleFactor==2 means 
            the classical pyramid, where each next level has 4x less pixels 
            than the previous, but such a big scale factor will degrade 
            feature matching scores dramatically. On the other hand, too 
            close to 1 scale factor will mean that to cover certain scale 
            range you will need more pyramid levels and so the speed will 
            suffer.*/
            SLfloat scaleFactor = 1.2f;
            
            /* The number of pyramid levels. The smallest level will have linear
            size equal to input_image_linear_size/pow(scaleFactor, nlevels).*/
            SLint nlevels = 8;
            
            /* This is size of the border where the features are not detected. 
            It should roughly match the patchSize parameter.*/
            SLint edgeThreshold = 31;
            
            // It should be 0 in the current implementation.
            SLint firstLevel = 0;
            
            /* The number of points that produce each element of the oriented 
            BRIEF descriptor. The default value 2 means the BRIEF where we take 
            a random point pair and compare their brightnesses, so we get 0/1 
            response. Other possible values are 3 and 4. For example, 3 means 
            that we take 3 random points (of course, those point coordinates 
            are random, but they are generated from the pre-defined seed, so 
            each element of BRIEF descriptor is computed deterministically from 
            the pixel rectangle), find point of maximum brightness and output 
            index of the winner (0, 1 or 2). Such output will occupy 2 bits, 
            and therefore it will need a special variant of Hamming distance,
            denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 
            random points to compute each bin (that will also occupy 2 bits 
            with possible values 0, 1, 2 or 3).*/
            SLint WTA_K = 2;
            
            /* The default HARRIS_SCORE means that Harris algorithm is used to 
            rank features (the score is written to KeyPoint::score and is used to 
            retain best nfeatures features); FAST_SCORE is alternative value of 
            the parameter that produces slightly less stable keypoints, but it 
            is a little faster to compute.*/
            int scoreType = ORB::HARRIS_SCORE;
            
            /* size of the patch used by the oriented BRIEF descriptor. Of 
            course, on smaller pyramid layers the perceived image area covered 
            by a feature will be larger.*/
            int patchSize = 31;
            int fastThreshold = 20;

            detector = ORB::create(nFeatures, 
                                   scaleFactor, 
                                   nlevels, 
                                   edgeThreshold, 
                                   firstLevel, 
                                   WTA_K, 
                                   scoreType, 
                                   patchSize, 
                                   fastThreshold);
        }
        break;

        case FT_SURF:
            _map.minHessian = 400.0f;
            detector = SURF::create(_map.minHessian);
            break;
    }

    //set type
    _map.type = type;

    ///////////////////////////////////////////////////////////////////////////////
    detector->detectAndCompute(imageGray, Mat(), _map.keypoints, _map.descriptors);
    ///////////////////////////////////////////////////////////////////////////////

    //copy points to new array
    SLCVVPoint2f imagePts;
    for(auto& keyPt : _map.keypoints)
        imagePts.push_back(keyPt.pt);

    if(imagePts.size())
    {
        //subpixels accuracy
        cornerSubPix(imageGray, 
                     imagePts, 
                     Size(11,11),
                     Size(-1,-1), 
                     TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1));
    }

    //calculate scale factor
    float refWidthMM = 1.0f;
    if(_refWidthStr.str().size() > 0)
    {
        float refWidthMM = std::stof(_refWidthStr.str());
        if(refWidthMM == 0.0f)
            cout << "Division by zero not possible. No scale factor applied." << endl;
    }
    else
        cout << "No scale factor applied." << endl;

    //extract and scale point positions
    for(size_t i = 0; i < _map.keypoints.size(); ++i)
    {
        //SLCVPoint2f pt =  _map.keypoints[i].pt / _map.scaleFactorPixPerMM;
        SLCVPoint2f pt =  imagePts[i];
        pt.y = - (pt.y - 480.0f);
        pt /= _map.scaleFactorPixPerMM;

        _map.pts.push_back(pt);
    }

    //save image
    _map.image = imageGray;

    //save to file
    _map.saveToFile(filename);
}
//-----------------------------------------------------------------------------
void SLCVFeatureMapper2D::clear()
{
    _state = IDLE;
    _refWidthStr.str(std::string());
    _refWidthStr.clear();
}
//-----------------------------------------------------------------------------
void SLCVFeatureMapper2D::removeLastDigit()
{
    string content = _refWidthStr.str();
    if(content.size())
    {
        content.pop_back();//.erase(content.end());
        _refWidthStr.clear();
        _refWidthStr.str(content);
    }
}
//-----------------------------------------------------------------------------
