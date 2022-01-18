//#############################################################################
//  File:      SLVRRenderModel.h
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLVRRENDERMODEL_H
#define SLPROJECT_SLVRRENDERMODEL_H

#include <openvr.h>

#include <SL.h>
#include <SLNode.h>
#include <SLAssetManager.h>
#include <SLGLTexture.h>

#include <vr/SLVR.h>

//-----------------------------------------------------------------------------
//! SLVRRenderModel represents the model of a tracked device that can be rendered in the scene
/*! The class is instantiated when calling loadRenderModel on an SLVRTrackedDevice.
 * Internally, the method "load" will be called, which loads the geometry and the texture from
 * the OpenVR API and creates an SLNode which can be attached to the scene graph.
 * The object matrix of the node will be automatically updated when calling SLVRSystem::update.
 */
class SLVRRenderModel
{
    friend class SLVRTrackedDevice;

public:
    // Getters
    SLNode* node() { return _node; };

private:
    void                load(const SLstring& name, SLAssetManager* assetManager);
    static SLGLTexture* loadTexture(vr::TextureID_t id, SLAssetManager* assetManager);
    static void         copyGeometryToMesh(vr::RenderModel_t* renderModel, SLMesh* mesh);

    SLNode* _node = nullptr;
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_SLVRRENDERMODEL_H
