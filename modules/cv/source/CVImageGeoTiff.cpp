//#############################################################################
//  File:      cv/CVImageGeoTiff.cpp
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <CVImageGeoTiff.h>
#include <json.hpp>
#include <Utils.h>
#include <SL.h>

using json = nlohmann::json;

//-----------------------------------------------------------------------------
CVImageGeoTiff::CVImageGeoTiff()
{
    _noDataValue = 0.0;
}
//-----------------------------------------------------------------------------
CVImageGeoTiff::~CVImageGeoTiff()
{
    clearData();
}
//-----------------------------------------------------------------------------
//! Loads a GEOTiff file into the OpenCV image matrix
void CVImageGeoTiff::loadGeoTiff(const string& geoTiffFile)
{
#ifndef __EMSCRIPTEN__
    string msg;

    // check if the GEOTiff file exists
    if (!Utils::fileExists(geoTiffFile))
    {
        msg = "CVImageGeoTiff::loadGeoTiff: File not found: " + geoTiffFile;
        throw std::runtime_error(msg.c_str());
    }

    // check if the GEOTiff json file exists
    string jsonFileName = Utils::getPath(geoTiffFile) +
                          Utils::getFileNameWOExt(geoTiffFile) +
                          ".json";
    if (!Utils::fileExists(jsonFileName))
    {
        msg = "CVImageGeoTiff::loadGeoTiff: JSON File not found: " + jsonFileName;
        msg += "\nA GEOTiff file must have a JSON file aside with the same name.";
        msg += "\nYou can generate this JSON file with the tool gdalinfo.";
        throw std::runtime_error(msg.c_str());
    }

    // Read the geo tiff image with OpenCV
    cv::Mat imgGeoTiff = cv::imread(geoTiffFile,
                                    cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH);

    if (imgGeoTiff.type() != CV_32FC1)
        throw std::runtime_error("GEOTiff image must be of 32-bit float type.");

    // Read the JSON file
    std::ifstream  jsonFile(jsonFileName);
    json           jsonData;
    string         description;
    string         geocsc;
    vector<int>    size;
    vector<double> upperLeft;
    vector<double> lowerRight;

    // Reading values from json
    try
    {
        jsonFile >> jsonData;
        description  = jsonData["description"].get<string>();
        geocsc       = jsonData["coordinateSystem"]["wkt"].get<string>();
        size         = jsonData["size"].get<vector<int>>();
        upperLeft    = jsonData["cornerCoordinates"]["upperLeft"].get<vector<double>>();
        lowerRight   = jsonData["cornerCoordinates"]["lowerRight"].get<vector<double>>();
        _noDataValue = jsonData["bands"][0]["noDataValue"].get<double>();
    }
    catch (json::exception& e)
    {
        msg = "Error reading JSON-File: " + jsonFileName;
        msg += "\nException: ";
        msg += e.what();
        throw std::runtime_error(msg.c_str());
    }

    // Check some correspondences between image file an json file
    if (size.size() < 2 || size[0] != imgGeoTiff.cols || size[1] != imgGeoTiff.rows)
    {
        msg = "Mismatch between geotiff image size and size json tag:";
        msg += "\nGEOTiff image width : " + to_string(imgGeoTiff.cols);
        msg += "\nGEOTiff image height: " + to_string(imgGeoTiff.rows);
        msg += "\nJSON Size tag[0]    : " + to_string(size[0]);
        msg += "\nJSON Size tag[1]    : " + to_string(size[1]);
        throw std::runtime_error(msg.c_str());
    }

    if (!Utils::containsString(geocsc, "WGS 84") &&
        !Utils::containsString(geocsc, "WGS_1984"))
    {
        msg = "GeoTiff file seams not have WGS84 coordinates.";
        throw std::runtime_error(msg.c_str());
    }

    _cvMat  = imgGeoTiff.clone();
    _format = cvType2glPixelFormat(imgGeoTiff.type());

    _upperleftLatLonAlt[0]  = upperLeft[1];           // We store first latitude in degrees! (N)
    _upperleftLatLonAlt[1]  = upperLeft[0];           // and then longitude in degrees (W)
    _upperleftLatLonAlt[2]  = _cvMat.at<float>(0, 0); // and then altitude in m from the image
    _lowerRightLatLonAlt[0] = lowerRight[1];          // we store first latitude in degrees! (S)
    _lowerRightLatLonAlt[1] = lowerRight[0];          // and then longitude in degrees (E)
    _lowerRightLatLonAlt[2] = _cvMat.at<float>(_cvMat.rows - 1, _cvMat.cols - 1);
#endif
}
//-----------------------------------------------------------------------------
//! Returns the altitude in m at the given position in WGS84 latitude-longitude
float CVImageGeoTiff::getAltitudeAtLatLon(double latDEG,
                                          double lonDEG) const
{
    double dLatDEG   = _upperleftLatLonAlt[0] - _lowerRightLatLonAlt[0];
    double dLonDEG   = _lowerRightLatLonAlt[1] - _upperleftLatLonAlt[1];
    double latPerPix = dLatDEG / (double)_cvMat.rows;
    double lonPerPix = dLonDEG / (double)_cvMat.cols;

    double offsetLat = latDEG - _lowerRightLatLonAlt[0];
    double offsetLon = lonDEG - _upperleftLatLonAlt[1];

    double pixPosLat = offsetLat / latPerPix; // pixels from bottom
    double pixPosLon = offsetLon / lonPerPix; // pixels from left

    // pixels are top-left coordinates in OpenCV
    pixPosLat = _cvMat.rows - pixPosLat;

    if (pixPosLat < 0.0 || pixPosLat > _cvMat.rows - 1.0)
    {
        SL_LOG("Invalid pixPosLat %3.2f", pixPosLat);
        pixPosLat = 0;
    }
    if (pixPosLon < 0.0 || pixPosLon > _cvMat.cols - 1.0)
    {
        SL_LOG("Invalid pixPosLon %3.2f", pixPosLon);
        pixPosLon = 0;
    }

    // get subpixel accurate interpolated height value
    cv::Point2f pt((float)pixPosLon, (float)pixPosLat);
    cv::Mat     patch;
    cv::getRectSubPix(_cvMat, cv::Size(1, 1), pt, patch);

    float heightMatPix    = _cvMat.at<float>((int)pixPosLat, (int)pixPosLon);
    float heightMatSubPix = patch.at<float>(0, 0);
    return heightMatSubPix;
}
//-----------------------------------------------------------------------------
