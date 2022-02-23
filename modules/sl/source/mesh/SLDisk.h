//#############################################################################
//  File:      SLDisk.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLDISK_h
#define SLDISK_h

#include <SLRevolver.h>

//-----------------------------------------------------------------------------
//! SLDisk creates a disk mesh based on SLRevolver
class SLDisk : public SLRevolver
{
public:
    SLDisk(SLAssetManager* assetMgr,
           SLfloat         radius      = 1.0f,
           const SLVec3f&  revolveAxis = SLVec3f::AXISY,
           SLuint          slices      = 36,
           SLbool          doubleSided = false,
           SLstring        name        = "disk mesh",
           SLMaterial*     mat         = nullptr);

    // Getters
    SLfloat radius() { return _radius; }

protected:
    SLfloat _radius;      //!< radius of cone
    SLbool  _doubleSided; //!< flag if disk has two sides
};
//-----------------------------------------------------------------------------
#endif // SLDISK_h
