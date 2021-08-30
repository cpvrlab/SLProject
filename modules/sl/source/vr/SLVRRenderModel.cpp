//#############################################################################
//  File:      SLVRRenderModel.cpp
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <vr/SLVRRenderModel.h>

SLVRRenderModel::SLVRRenderModel()
{
}

/*! Loads the render model from disk
 * The method walks through the following steps:
 * 1. Load the geometry
 * 2. Load the texture
 * 3. Combine the geometry and texture into a SLMesh
 * 4. Attach the mesh to a newly created SLNode
 * @param name The OpenVR name of the render model to be loaded
 * @param assetManager The asset manager that will own all the assets (the mesh, the texture, etc.)
 */
void SLVRRenderModel::load(const SLstring& name, SLAssetManager* assetManager)
{
    VR_LOG("Loading render model \"" << name << "\"...")

    // Load the render model
    vr::RenderModel_t* renderModel;
    while (true)
    {
        vr::EVRRenderModelError error = vr::VRRenderModels()->LoadRenderModel_Async(name.c_str(), &renderModel);
        if (error != vr::EVRRenderModelError::VRRenderModelError_Loading)
            break;
    }

    // Load the texture
    SLGLTexture* texture = loadTexture(renderModel->diffuseTextureId, assetManager);

    // Create a SLMesh for the render model
    SLMesh* mesh = new SLMesh(assetManager, name);
    mesh->primitive(PT_triangles);
    mesh->mat(new SLMaterial(assetManager, (name + " Material").c_str(), texture));
    copyRenderModelGeometryToMesh(renderModel, mesh);

    // The render model is no longer used, so we delete it
    vr::VRRenderModels()->FreeRenderModel(renderModel);

    _node = new SLNode(mesh, name);
}

/*! Loads the render model texture from disk
 * After loading, the image data gets converted from the RGBA format to the RGB format
 * The image is then converted to a CVImage (RGB format) and attached to a SLGLTexture
 * @param id The OpenVR ID of the texture to be loaded
 * @param assetManager The asset manager that will own the texture
 * @return The loaded texture as a SLGLTexture
 */
SLGLTexture* SLVRRenderModel::loadTexture(vr::TextureID_t id, SLAssetManager* assetManager)
{
    // Load the texture
    vr::RenderModel_TextureMap_t* renderModelTextureMap;
    while (true)
    {
        vr::EVRRenderModelError error = vr::VRRenderModels()->LoadTexture_Async(id, &renderModelTextureMap);
        if (error != vr::EVRRenderModelError::VRRenderModelError_Loading)
            break;
    }

    // Convert the RGBA format to the RGB format
    const uint8_t* sourceData = renderModelTextureMap->rubTextureMapData;
    unsigned char* targetData = new unsigned char[renderModelTextureMap->unWidth * renderModelTextureMap->unHeight * 3];

    for (int i = 0; i < renderModelTextureMap->unWidth * renderModelTextureMap->unHeight; i++)
    {
        targetData[i * 3 + 0] = sourceData[i * 4 + 0];
        targetData[i * 3 + 1] = sourceData[i * 4 + 1];
        targetData[i * 3 + 2] = sourceData[i * 4 + 2];
    }

    // Store the image data in a CVImage
    CVImage* image = new CVImage();
    image->load(renderModelTextureMap->unWidth,
                renderModelTextureMap->unHeight,
                CVPixFormat::PF_rgb,
                CVPixFormat::PF_rgb,
                targetData,
                true,
                false);

    // Create the texture from the CVImage
    SLGLTexture* texture = new SLGLTexture(assetManager, GL_LINEAR, GL_LINEAR, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER);
    texture->images().push_back(image);
    texture->textureSize(image->width(), image->height());
    texture->texType(TT_diffuse);

    // The OpenVR texture map is no longer used, so we delete it
    vr::VRRenderModels()->FreeTexture(renderModelTextureMap);

    return texture;
}

/*! Copies the OpenVR render model data (vertices, UVs, normals, indices) to a SLMesh
 * @param renderModel The render model that contains the source data
 * @param mesh The mesh that the data will be copied to
 */
void SLVRRenderModel::copyRenderModelGeometryToMesh(vr::RenderModel_t* renderModel, SLMesh* mesh)
{
    // Copy the vertices and normals
    for (uint32_t i = 0; i < renderModel->unVertexCount; i++)
    {
        vr::RenderModel_Vertex_t openVRVertex = renderModel->rVertexData[i];
        SLVec3f                  position     = SLVec3f(openVRVertex.vPosition.v);
        SLVec3f                  normal       = SLVec3f(openVRVertex.vNormal.v);
        SLVec2f                  uv           = SLVec2f(openVRVertex.rfTextureCoord);

        mesh->P.push_back(position);

        // The OpenVR API can apparently return zero normals for the entire model
        // We don't add these normals to the vector, so it will be empty at the end
        // The SLMesh will then see that the vector is empty and thus calculate the normals on its own
        if (!normal.isZero())
            mesh->N.push_back(normal);

        mesh->UV1.push_back(uv);
    }

    // Copy the indices
    for (uint32_t i = 0; i < renderModel->unTriangleCount * 3; i++)
    {
        uint16_t openVRIndex = renderModel->rIndexData[i];
        mesh->I16.push_back(openVRIndex);
    }
}