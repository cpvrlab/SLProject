//#############################################################################
//  File:      cv/CVImageGeoTiff.h
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVIMAGEGEOTIFF_H
#define CVIMAGEGEOTIFF_H

#include <CVImage.h>

//-----------------------------------------------------------------------------
//! Encapsulates a GEOTiff images with geo referenced meta information
/*! A GEOTiff image can hold per pixel the height information of a "rectangular"
 area on earth that is defined with upper-left and lower-left corner in longitude
 and latitude coordinates. With the loadGeoTiff function only GeoTiffs with
 WGS84 (EPSG 4326) coordinates can be loaded.
 GeoTiff with other coordinate reference systems e.g. the Swiss LV95 can be
 converted first in tools such as QGIS. Because we can not load the meta
 information with OpenCV we have to store them in a separate json file with
 the same name. They are generated with a tool that comes with QGIS as follows:
 gdalinfo -json DTM-Aventicum-WGS84.tif > DTM-Aventicum-WGS84.json
 */
class CVImageGeoTiff : public CVImage
{
public:
    CVImageGeoTiff();
    ~CVImageGeoTiff();

    void    loadGeoTiff(const string& filename);
    CVVec3d upperLeftLatLonAlt() const { return _upperleftLatLonAlt; }
    CVVec3d lowerRightLatLonAlt() const { return _lowerRightLatLonAlt; }
    float   getAltitudeAtLatLon(double lat, double lon) const;

private:
    CVVec3d _upperleftLatLonAlt;  //! Upper-left corner of DEM in WGS84 coords
    CVVec3d _lowerRightLatLonAlt; //! Lower-right corner of DEM in WGS84 coords
    double  _noDataValue;         //! double pixel value that stands for no data
};
//-----------------------------------------------------------------------------
#endif
