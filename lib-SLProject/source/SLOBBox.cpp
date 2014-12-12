//#############################################################################
//  File:      SLOBBox.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLOBBox.h>

/*
    TODO:   
        1. Implement SLOBBox functions
        2. Add a seperate map of SLOBBoxes to SLSkeleton
        3. Generate an SLOBBox in the importer for each bone
            if it already exists merge it to the existing one (happens when multiple meshes use the same skeleton)
        4. After animating a skeleton calculate the min and max points from all the OBBs by transforming with the joint matrix
            > this means we need to have two states saved in the OBBs a transformed state and an original state
        5. Now all we have to do is the same as in the CPU skinning, mark all nodes containing the animated mesh
            as dirty and return the skeletons min and max point when requested to (in SLMesh::buildAABB)
*/

SLOBBox::SLOBBox()
{

}

void SLOBBox::build(const SLVec3f* points, SLuint numP)
{

}
void SLOBBox::merge(const SLOBBox& other)
{

}

void SLOBBox::draw(const SLCol3f color)
{

}
void SLOBBox::generateVBO()
{

}