//#############################################################################
//  File:      CV/CVImageGeoTiff.cpp
//  Author:    Marcus Hudritsch
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <CVImageGeoTiff.h>
#include <nlohmann/json.hpp>
#include <utils.h>

using json = nlohmann::json;

//-----------------------------------------------------------------------------
CVImageGeoTiff::~CVImageGeoTiff()
{
    clearData();
}
//-----------------------------------------------------------------------------
//! Loads a GEOTiff file into the OpenCV image matrix
void CVImageGeoTiff::loadGeoTiff(const string& appTag,
                                 const string& geoTiffFile)
{
    string msg;

    // check if the GEOTiff file exists
    if (!Utils::fileExists(geoTiffFile))
    {
        msg = "CVImageGeoTiff::loadGeoTiff: File not found: " + geoTiffFile;
        Utils::exitMsg(appTag.c_str(), msg.c_str(), __LINE__, __FILE__);
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
        Utils::exitMsg(appTag.c_str(), msg.c_str(), __LINE__, __FILE__);
    }

    // Read the geo tiff image with OpenCV
    cv::Mat imgGeoTiff = cv::imread(geoTiffFile, cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH);

    // Read the JSON file
    ifstream       jsonFile(jsonFileName);
    json           json;
    string         description;
    string         geocsc;
    vector<int>    size;
    vector<double> upperLeft;
    vector<double> lowerRight;

    try // Reading values from json
    {
        jsonFile >> json;
        cout << json.dump(4);
        description = json["description"].get<string>();
        geocsc      = json["coordinateSystem"]["wkt"].get<string>();
        size        = json["size"].get<vector<int>>();
        upperLeft   = json["cornerCoordinates"]["upperLeft"].get<vector<double>>();
        lowerRight  = json["cornerCoordinates"]["lowerRight"].get<vector<double>>();
    }
    catch (json::exception& e)
    {
        msg = "Error reading JSON-File: " + jsonFileName;
        msg += "\nException: ";
        msg += e.what();
        Utils::exitMsg(appTag.c_str(), msg.c_str(), __LINE__, __FILE__);
    }

    // Check some correspondences between image file an json file
    if (size.size() < 2 || size[0] != imgGeoTiff.cols || size[1] != imgGeoTiff.rows)
    {
        msg = "Mismatch between geotiff image size and size json tag:";
        msg += "\nGEOTiff image width : " + to_string(imgGeoTiff.cols);
        msg += "\nGEOTiff image height: " + to_string(imgGeoTiff.rows);
        msg += "\nJSON Size tag[0]    : " + to_string(size[0]);
        msg += "\nJSON Size tag[1]    : " + to_string(size[1]);
        Utils::exitMsg(appTag.c_str(), msg.c_str(), __LINE__, __FILE__);
    }

    if (!Utils::containsString(geocsc, "WGS 84") &&
        !Utils::containsString(geocsc, "WGS_1984"))
    {
        msg = "GeoTiff file seams not have WGS84 coordinates.";
        Utils::exitMsg(appTag.c_str(), msg.c_str(), __LINE__, __FILE__);
    }

    _cvMat = imgGeoTiff.clone();
    _format = cv2glPixelFormat(imgGeoTiff.type());
    _upperleftLLA[0] = upperLeft[0];
    _upperleftLLA[1] = upperLeft[1];
    _lowerRightLLA[0] = lowerRight[0];
    _lowerRightLLA[1] = lowerRight[1];
}
//-----------------------------------------------------------------------------