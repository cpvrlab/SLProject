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
void CVImageGeoTiff::loadGeoTiff(const string& geoTiffFile)
{
    // check if the GEOTiff file exists
    if (!Utils::fileExists(geoTiffFile))
    {
        string msg = "CVImageGeoTiff::loadGeoTiff: File not found: " + geoTiffFile;
        Utils::log("SLProject", msg.c_str());
        exit(-1);
    }

    // check if the GEOTiff json file exists
    string jsonFileName = Utils::getPath(geoTiffFile) +
                          Utils::getFileNameWOExt(geoTiffFile) +
                          ".json";
    if (!Utils::fileExists(jsonFileName))
    {
        string msg = "CVImageGeoTiff::loadGeoTiff: JSON File not found: " + jsonFileName;
        Utils::log("SLProject", msg.c_str());
        exit(-1);
    }

    _cvMat = cv::imread(geoTiffFile, cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH);

    // Read the JSON file
    ifstream jsonFile(jsonFileName);
    json     geoTiffJson;
    try
    {
        jsonFile >> geoTiffJson;
    }
    catch (exception& e)
    {
        Utils::log("SLProject", "Error reading JSON-File: %s", jsonFileName.c_str());
        exit(-1);
    }
}
//-----------------------------------------------------------------------------