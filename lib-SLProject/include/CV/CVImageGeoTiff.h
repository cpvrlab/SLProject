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
//!
/*!

*/
class CVImageGeoTiff : CVImage
{
public:
    CVImageGeoTiff() = default;
    ~CVImageGeoTiff();

    void loadGeoTiff(const string& filename);
};
//-----------------------------------------------------------------------------
#endif