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
 If the elevation data is available in ESRI-ASCII format you have to download
 them first from the BFH server (only within the BFH network possible):
 smb://bfh.ch/data/LFE/BFH/Geodata/swisstopo/meta
 P:\LFE\BFH\Geodata\swisstopo\meta
 The ASCII patches are in DISK3/swissALTI3D
 with cp SWISSALTI3D_0.5_ESRIASCIIGRID_CHLV95_LN02_2585_1221.zip ~/Downloads
 The patch number can be found on:
 https://map.geo.admin.ch/?lang=de&topic=ech&bgLayer=ch.swisstopo.pixelkarte-farbe&layers=ch.swisstopo.images-swissimage-dop10.metadata&E=2752545.83&N=1178500.73&zoom=1
 */
class CVImageGeoTiff : public CVImage
{
public:
    CVImageGeoTiff();
    ~CVImageGeoTiff();

    void    loadGeoTiff(const string& filename);
    CVVec3d upperLeftLatLonAlt() { return _upperleftLatLonAlt; }
    CVVec3d lowerRightLatLonAlt() { return _lowerRightLatLonAlt; }
    float   getAltitudeAtLatLon(double lat, double lon);

private:
    CVVec3d _upperleftLatLonAlt;  //! Upper-left corner of DEM in WGS84 coords
    CVVec3d _lowerRightLatLonAlt; //! Lower-right corner of DEM in WGS84 coords
    double  _noDataValue;         //! double pixel value that stands for no data
};
//-----------------------------------------------------------------------------
#endif
