//#############################################################################
//  File:      AR2DMapper.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef AR2DMAPPER_H
#define AR2DMAPPER_H

#include <opencv2/core.hpp>
#include "opencv2/highgui.hpp"

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

    void saveToFile( std::string dir, std::string filename )
    {
        cv::Mat points2d(pts);
        std::string typeName = mapType(type);
        cv::FileStorage storage( dir + filename + ".yml", cv::FileStorage::WRITE);
        storage << "points2d" << points2d;
        storage << "descriptors" << descriptors;

        storage << "scaleFactor" << scaleFactorPixPerMM;
        storage << "transVect" << transVect;
        storage << "rotVect" << rotVect;
        storage << "type" << typeName;
        storage << "minHessian" << minHessian;
        storage << "keypoints" << keypoints;

        storage.release();

        cv::imwrite( dir + filename + ".png", image );
    }

    void loadFromFile( std::string dir, std::string filename )
    {
        cv::Mat points2d;
        std::string typeName;
        cv::FileStorage storage( dir + filename + ".yml", cv::FileStorage::READ);
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

        image = cv::imread( dir + filename + ".png" );
    }

    //coordinate positions
    std::vector<cv::Point2d> pts;
    //descriptors
    cv::Mat descriptors;
    //scale factor
    float scaleFactorPixPerMM;
    //translation vector
    cv::Mat transVect;
    //rotation vector
    cv::Mat rotVect;
    //feature type
    ARFeatureType type;
    //minHessian for surf
    float minHessian;

    //debug
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;

private:

    std::string mapType( ARFeatureType type )
    {
        switch(type) {
        case AR_ORB:
            return "AR_ORB";
        case AR_SURF:
            return "AR_SURF";

        return "";
        }
    }
//-----------------------------------------------------------------------------
    ARFeatureType mapType( std::string name )
    {
        if(name == "AR_ORB")
            return AR_ORB;
        else if(name == "AR_SURF")
            return AR_SURF;
    }
};

//-----------------------------------------------------------------------------
class AR2DMapper
{
public:
    enum Mapper2DState { IDLE, LINE_INPUT, CAPTURE };
    //constructor
    AR2DMapper();
    //create a mapping of type AR2DMap
    void createMap( cv::Mat image, float offsetXMM, float offsetYMM,
                    std::string dir, std::string filename, AR2DMap::ARFeatureType type );

    void clear();
    bool stateLineInput() { return _state == LINE_INPUT; }
    bool stateCapture() { return _state == CAPTURE; }
    bool stateIdle() { return _state == IDLE; }

    void addDigit( std::string str ) { _refWidthStr << str; }
    std::string getCurrentRefWidthStr() { return _refWidthStr.str(); }
    void setState( Mapper2DState state ) { _state = state; }

    void removeLastDigit();

private:
    //input image
    cv::Mat _image;
    //created
    AR2DMap _map;
    //current state
    Mapper2DState _state;

    //reference width in meter
    std::stringstream _refWidthStr;
};
//-----------------------------------------------------------------------------

#endif // AR2DMAPPER_H
