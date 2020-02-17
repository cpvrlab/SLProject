//#############################################################################
//  File:      CV/CVImageGeoTiff.h
//  Author:    Marcus Hudritsch
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVIMAGEGEOTIFF_H
#define CVIMAGEGEOTIFF_H

#include <CVImage.h>

//-----------------------------------------------------------------------------
//! Encapsulates a GEOTiff images with geo referenced meta informations
/*! A GEOTiff image can hold per pixel the height information of a "rectangular"
 area on earth that is defined with upper-left and lower-left corner in longitude
 and lattitude coordinates. With the loadGeoTiff function only GeoTiffs with
 WGS84 coordinates can be loaded. GeoTiff with other coordinate reference systems
 e.g. the Swiss LV95 can be converted first in tools such as QGIS. Because we
 can not load the meta information with OpenCV we have to store them in a
 separate json file with the same name. They are generated with the gdaltool as
 follows: gdaltool geotifffile.tif -json > geotifffile.json
*/
class CVImageGeoTiff : public CVImage
{
public:
    CVImageGeoTiff();
    ~CVImageGeoTiff();

    void    loadGeoTiff(const string& appTag, const string& filename);
    CVVec3d upperLeftLLA() { return _upperleftLLA; }
    CVVec3d lowerRightLLA() { return _lowerRightLLA; }
    double  getHeightAtLatLon(float lat, float lon);

private:
    CVVec3d _upperleftLLA;  //! Upper-left corner of DEM in WGS84 coords
    CVVec3d _lowerRightLLA; //! Lower-right corner of DEM in WGS84 coords
    double  _noDataValue;   //! double pixel value that stands for no data
};
//-----------------------------------------------------------------------------
#endif