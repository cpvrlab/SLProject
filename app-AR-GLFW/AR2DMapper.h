//#############################################################################
//  File:      AR2DMapper.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef AR2DMAPPER_H
#define AR2DMAPPER_H

#include <SLCVCalibration.h>

#include <opencv2/core.hpp>
#include "opencv2/highgui.hpp"

using namespace std;

//-----------------------------------------------------------------------------
class AR2DMap
{
public:
    //enum for feature type
    enum ARFeatureType { AR_ORB, AR_SURF, AR_NO_TYPE };

    AR2DMap() :
        scaleFactorPixPerMM(1.0),
        type(AR_NO_TYPE),
        minHessian(400.0f)
    {}

    void saveToFile(std::string dir, std::string filename)
    {
        cv::Mat points2d(pts);
        std::string typeName = mapType(type);
        cv::FileStorage storage(dir + filename + ".yml", cv::FileStorage::WRITE);
        storage << "points2d" << points2d;
        storage << "descriptors" << descriptors;
        storage << "scaleFactor" << scaleFactorPixPerMM;
        storage << "transVect" << transVect;
        storage << "rotVect" << rotVect;
        storage << "type" << typeName;
        storage << "minHessian" << minHessian;
        storage << "keypoints" << keypoints;

        storage.release();

        cv::imwrite(dir + filename + ".png", image);
    }

    void loadFromFile(std::string filename)
    {
        cv::Mat points2d;
        std::string typeName;
        cv::FileStorage storage(SLCVCalibration::defaultPath + filename + ".yml", 
                                cv::FileStorage::READ);
        storage["points2d"] >> points2d;
        storage["descriptors"] >> descriptors;
        storage["scaleFactor"] >> scaleFactorPixPerMM;
        storage["transVect"] >> transVect;
        storage["rotVect"] >> rotVect;
        storage["type"] >> typeName;
        storage["minHessian"] >> minHessian;
        storage["keypoints"] >> keypoints;

        points2d.copyTo(pts);
        type = mapType(typeName);

        image = cv::imread(SLCVCalibration::defaultPath + filename + ".png");
    }


    vector<cv::Point2d> pts;                //!< coordinate positions
    cv::Mat             descriptors;        //!< descriptors
    float               scaleFactorPixPerMM;//!< scale factor
    cv::Mat             transVect;          //!< translation vector
    cv::Mat             rotVect;            //!< rotation vector
    ARFeatureType       type;               //!< feature type
    float               minHessian;         //!< minHessian for surf
    cv::Mat             image;              //!< debug image
    vector<cv::KeyPoint> keypoints;

private:

    string mapType(ARFeatureType type)
    {   switch(type)
        {   case AR_ORB:    return "AR_ORB";
            case AR_SURF:   return "AR_SURF";
            default:        return "";
        }
    }

    ARFeatureType mapType(string name)
    {   if (name=="AR_ORB") return AR_ORB; else
        if (name=="AR_SURF") return AR_SURF;
        return AR_NO_TYPE;
    }
};

//-----------------------------------------------------------------------------
class AR2DMapper
{
    public:
        enum Mapper2DState {IDLE, LINE_INPUT, CAPTURE};

                AR2DMapper();

        void    createMap       (cv::Mat image,
                                 float offsetXMM,
                                 float offsetYMM,
                                 string filename,
                                 AR2DMap::ARFeatureType type);

        void    clear           ();
        bool    stateIsLineInput() {return _state == LINE_INPUT;}
        bool    stateIsCapture  () {return _state == CAPTURE;}
        bool    stateIsIdle     () {return _state == IDLE;}

        void    addDigit        (string str) {_refWidthStr << str;}
        string  getRefWidthStr  () {return _refWidthStr.str();}
        void    state           (Mapper2DState newState) {_state = newState;}

        void    removeLastDigit ();

    private:
        cv::Mat         _image;         //!< input image
        AR2DMap         _map;           //!< created
        Mapper2DState   _state;         //!< current state
        stringstream    _refWidthStr;   //!< reference width in meter
};
//-----------------------------------------------------------------------------

#endif // AR2DMAPPER_H
