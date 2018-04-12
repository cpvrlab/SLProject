//#############################################################################
//  File:      SLSkybox
//  Author:    Marcus Hudritsch
//  Date:      December 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSKYBOX_H
#define SLSKYBOX_H

#include <SLEnums.h>
#include <SLNode.h>

class SLSceneView;

//-----------------------------------------------------------------------------
//! Skybox node class with a SLBox mesh
/*! The skybox instance is a node with a SLBox mesh with inwards pointing
normals. It gets drawn in SLSceneView::draw3DGL with frozen depth buffer and a
special cubemap shader. The box is allways with the active camera in its
center. It has to be created in SLScene::onLoad and assigned to the skybox
pointer of SLSceneView. See the Skybox shader example.
*/
class SLSkybox: public SLNode
{
    public:
                    SLSkybox        (SLstring name = "Default Skybox");
                    SLSkybox        (SLstring cubeMapXPos,
                                     SLstring cubeMapXNeg,
                                     SLstring cubeMapYPos,
                                     SLstring cubeMapYNeg,
                                     SLstring cubeMapZPos,
                                     SLstring cubeMapZNeg,
                                     SLstring name = "Default Skybox");
                   ~SLSkybox        (){;}
    
        SLCol4f     colorAtDir      (SLVec3f dir);
    
        void        drawAroundCamera (SLSceneView* sv);
};
//-----------------------------------------------------------------------------
#endif // #define SLSKYBOX_H
