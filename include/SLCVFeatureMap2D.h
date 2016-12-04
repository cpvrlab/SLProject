//#############################################################################
//  File:      SLCVFeatureMap2D.cpp
//  Author:    Michael Göttlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVFEATUREMAP2D_H
#define SLCVFEATUREMAP2D_H

/* 
If an application uses live video processing you have to define 
the preprocessor contant SL_HAS_OPENCV in the project settings.
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/
#ifdef SL_HAS_OPENCV

#include <SLCV.h>

//-----------------------------------------------------------------------------
class SLCVFeatureMap2D
{
public:

    SLCVFeatureMap2D() :
        scaleFactorPixPerMM(1.0),
        type(FT_ORB),
        minHessian(400.0f)
    {}

    void saveToFile(SLstring filename)
    {
        SLCVMat points2d(pts);
        SLstring typeName = mapType(type);
        cv::FileStorage storage(SL::configPath + filename + ".yml", 
                                SLCVFileStorage::WRITE);
        storage << "points2d" << points2d;
        storage << "descriptors" << descriptors;
        storage << "scaleFactor" << scaleFactorPixPerMM;
        storage << "transVect" << tVec;
        storage << "rotVect" << rVec;
        storage << "type" << typeName;
        storage << "minHessian" << minHessian;
        storage << "keypoints" << keypoints;
        storage.release();

        cv::imwrite(SL::configPath + filename + ".png", image);
    }

    void loadFromFile(std::string filename)
    {
        SLCVMat points2d;
        SLstring typeName;
        SLCVFileStorage storage(SL::configPath + filename + ".yml", 
                                SLCVFileStorage::READ);
        storage["points2d"]     >> points2d;
        storage["descriptors"]  >> descriptors;
        storage["scaleFactor"]  >> scaleFactorPixPerMM;
        storage["tVect"]        >> tVec;
        storage["rVect"]        >> rVec;
        storage["type"]         >> typeName;
        storage["minHessian"]   >> minHessian;
        storage["keypoints"]    >> keypoints;
        storage.release();

        points2d.copyTo(pts);
        type = mapType(typeName);

        image = cv::imread(SL::configPath + filename + ".png");
    }

    SLCVVPoint2d    pts;                //!< coordinate positions
    SLCVMat         descriptors;        //!< descriptors
    SLfloat         scaleFactorPixPerMM;//!< scale factor
    SLCVMat         tVec;               //!< translation vector
    SLCVMat         rVec;               //!< rotation vector
    SLCVFeatureType type;               //!< feature type
    SLfloat         minHessian;         //!< minHessian for surf
    SLCVMat         image;              //!< debug image
    SLCVVKeyPoint   keypoints;          //!< vector of key points

private:

    SLstring mapType(SLCVFeatureType type)
    {   switch(type)
        {   case FT_ORB:    return "FT_ORB";
            case FT_SURF:   return "FT_SURF";
            case FT_SIFT:   return "FT_SIFT";
            default:        SL_EXIT_MSG("Unknown feature type");
        }
        return "";
    }

    SLCVFeatureType mapType(string name)
    {   if (name=="FT_ORB") return FT_ORB; else
        if (name=="FT_SURF") return FT_SURF; else
        if (name=="FT_SIFT") return FT_SIFT; else
        SL_EXIT_MSG("Unknown feature type");
        return FT_ORB;
    }
};
//-----------------------------------------------------------------------------

#endif // SLCVFEATUREMAP2D_H
#endif // SL_HAS_OPENCV