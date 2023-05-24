//#############################################################################
//  File:      SLMesh.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLCompactGrid.h>
#include <SLNode.h>
#include <SLRay.h>
#include <SLRaytracer.h>
#include <SLSceneView.h>
#include <SLSkybox.h>
#include <SLMesh.h>
#include <SLAssetManager.h>
#include <Profiler.h>

using std::set;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <igl/remove_duplicate_vertices.h>
#include <igl/per_face_normals.h>
#include <igl/unique_edge_map.h>
#pragma clang diagnostic pop

//-----------------------------------------------------------------------------
/*!
 * Constructor for mesh objects.
 * Meshes can be used in multiple nodes (SLNode). Meshes can belong
 * therefore to the global assets such as meshes (SLMesh), materials
 * (SLMaterial), textures (SLGLTexture) and shader programs (SLGLProgram).
 * @param assetMgr Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param name Name of the mesh
 */
SLMesh::SLMesh(SLAssetManager* assetMgr, const SLstring& name) : SLObject(name)
{
    _primitive = PT_triangles;
    _mat       = nullptr;
    _matOut    = nullptr;
    _finalP    = &P;
    _finalN    = &N;
    minP.set(FLT_MAX, FLT_MAX, FLT_MAX);
    maxP.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    _skeleton               = nullptr;
    _isVolume               = true;    // is used for RT to decide inside/outside
    _accelStruct            = nullptr; // no initial acceleration structure
    _accelStructIsOutOfDate = true;
    _isSelected             = false;
    _edgeAngleDEG           = 30.0f;
    _edgeWidth              = 2.0f;
    _edgeColor              = SLCol4f::WHITE;
    _vertexPosEpsilon       = 0.001f;

    // Add this mesh to the global resource vector for deallocation
    if (assetMgr)
        assetMgr->meshes().push_back(this);
}
//-----------------------------------------------------------------------------
//! The destructor deletes everything by calling deleteData.
/*!
 * The destructor should be called by the owner of the mesh. If an asset manager
 * was passed in the constructor it will do it after scene destruction.
 * The material (SLMaterial) that the mesh uses will not be deallocated.
 */
SLMesh::~SLMesh()
{
    deleteData();
}
//-----------------------------------------------------------------------------
//! SLMesh::deleteData deletes all mesh data and vbo's
void SLMesh::deleteData()
{
    P.clear();
    N.clear();
    C.clear();
    T.clear();
    UV[0].clear();
    UV[1].clear();
    for (auto i : Ji) i.clear();
    Ji.clear();
    for (auto i : Jw) i.clear();
    Jw.clear();
    I16.clear();
    I32.clear();
    IS32.clear();
    IE16.clear();
    IE32.clear();

    _jointMatrices.clear();
    skinnedP.clear();
    skinnedN.clear();

    if (_accelStruct)
    {
        delete _accelStruct;
        _accelStruct = nullptr;
    }

    _vao.deleteGL();
    _vaoN.deleteGL();
    _vaoT.deleteGL();

#ifdef SL_HAS_OPTIX
    _vertexBuffer.free();
    _normalBuffer.free();
    _indexShortBuffer.free();
    _indexIntBuffer.free();
#endif
}
//-----------------------------------------------------------------------------
void SLMesh::deleteDataGpu()
{
    _vao.deleteGL();
    _vaoN.deleteGL();
    _vaoT.deleteGL();

#ifdef SL_HAS_OPTIX
    _vertexBuffer.free();
    _normalBuffer.free();
    _indexShortBuffer.free();
    _indexIntBuffer.free();
#endif
}
//-----------------------------------------------------------------------------
//! Deletes the rectangle selected vertices and the dependent triangles.
/*! The selection rectangle is defined in SLScene::selectRect and gets set and
 drawn in SLCamera::onMouseDown and SLCamera::onMouseMove. All vertices that
 are within the selectRect are listed in SLMesh::IS32. The selection evaluation
 is done during drawing in SLMesh::handleRectangleSelection and is only valid
 for the current frame. See also SLMesh::handleRectangleSelection.*/
void SLMesh::deleteSelected(SLNode* node)
{
    // Loop over all rectangle selected indexes in IS32
    for (SLulong i = 0; i < IS32.size(); ++i)
    {
        SLulong ixDel = IS32[i] - i;

        if (ixDel < P.size()) P.erase(P.begin() + ixDel);
        if (ixDel < N.size()) N.erase(N.begin() + ixDel);
        if (ixDel < C.size()) C.erase(C.begin() + ixDel);
        if (ixDel < T.size()) T.erase(T.begin() + ixDel);
        if (ixDel < UV[0].size()) UV[0].erase(UV[0].begin() + ixDel);
        if (ixDel < UV[1].size()) UV[1].erase(UV[1].begin() + ixDel);
        if (ixDel < Ji.size()) Ji.erase(Ji.begin() + ixDel);
        if (ixDel < Jw.size()) Jw.erase(Jw.begin() + ixDel);

        // Loop over all 16 bit triangles indexes
        if (!I16.empty())
        {
            SLVushort i16;
            // copy the triangle that do not contain the index to delete
            for (SLulong t = 0; t < I16.size(); t += 3)
            {
                if (I16[t] != ixDel &&
                    I16[t + 1] != ixDel &&
                    I16[t + 2] != ixDel)
                {
                    if (I16[t] < ixDel)
                        i16.push_back(I16[t]);
                    else
                        i16.push_back(I16[t] - 1);
                    if (I16[t + 1] < ixDel)
                        i16.push_back(I16[t + 1]);
                    else
                        i16.push_back(I16[t + 1] - 1);
                    if (I16[t + 2] < ixDel)
                        i16.push_back(I16[t + 2]);
                    else
                        i16.push_back(I16[t + 2] - 1);
                }
            }
            I16 = i16;
        }

        // Loop over all 32 bit triangles indexes
        if (!I32.empty())
        {
            SLVuint i32;
            // copy the triangle that do not contain the index to delete
            for (SLulong t = 0; t < I32.size(); t += 3)
            {
                if (I32[t] != ixDel &&
                    I32[t + 1] != ixDel &&
                    I32[t + 2] != ixDel)
                {
                    if (I32[t] < ixDel)
                        i32.push_back(I32[t]);
                    else
                        i32.push_back(I32[t] - 1);
                    if (I32[t + 1] < ixDel)
                        i32.push_back(I32[t + 1]);
                    else
                        i32.push_back(I32[t + 1] - 1);
                    if (I32[t + 2] < ixDel)
                        i32.push_back(I32[t + 2]);
                    else
                        i32.push_back(I32[t + 2] - 1);
                }
            }
            I32 = i32;
        }
    }

    deleteUnused();

    calcNormals();

    // build tangents for bump mapping
    if (mat()->needsTangents() && !UV[0].empty() && T.empty())
        calcTangents();

    // delete vertex array object so it gets regenerated
    _vao.clearAttribs();
    _vaoS.deleteGL();
    _vaoN.deleteGL();
    _vaoT.deleteGL();

    // delete the selection indexes
    IS32.clear();

    // flag aabb and aceleration structure to be updated
    node->needAABBUpdate();
    _accelStructIsOutOfDate = true;
}
//-----------------------------------------------------------------------------
//! Deletes unused vertices (= vertices that are not indexed in I16 or I32)
void SLMesh::deleteUnused()
{
    // SLPoints have no indexes, so nothing to remove
    if (I16.empty() && I32.empty())
        return;

    // A boolean for each vertex to flag it as used or not
    SLVbool used(P.size());
    for (auto&& u : used)
        u = false;

    // Loop over all indexes and mark them as used
    for (unsigned short i : I16)
        used[i] = true;

    for (unsigned int i : I32)
        used[i] = true;

    SLuint unused = 0;
    for (SLulong u = 0; u < used.size(); ++u)
    {
        if (!used[u])
        {
            unused++;
            SLulong ixDel = u - (unused - 1);

            if (ixDel < P.size()) P.erase(P.begin() + ixDel);
            if (ixDel < N.size()) N.erase(N.begin() + ixDel);
            if (ixDel < C.size()) C.erase(C.begin() + ixDel);
            if (ixDel < T.size()) T.erase(T.begin() + ixDel);
            if (ixDel < UV[0].size()) UV[0].erase(UV[0].begin() + ixDel);
            if (ixDel < UV[1].size()) UV[1].erase(UV[1].begin() + ixDel);
            if (ixDel < Ji.size()) Ji.erase(Ji.begin() + ixDel);
            if (ixDel < Jw.size()) Jw.erase(Jw.begin() + ixDel);

            // decrease the indexes smaller than the deleted on
            for (unsigned short& i : I16)
            {
                if (i > ixDel)
                    i--;
            }

            for (unsigned int& i : I32)
            {
                if (i > ixDel)
                    i--;
            }
        }
    }
}
//-----------------------------------------------------------------------------
//! SLMesh::shapeInit sets the transparency flag of the AABB
void SLMesh::init(SLNode* node)
{
    // Check data
    SLstring msg;
    if (P.empty())
        msg = "No vertex positions (P)\n";
    if (_primitive != PT_points && I16.empty() && I32.empty())
        msg += "No vertex indices (I16 or I32)\n";
    if (msg.length() > 0)
        SL_EXIT_MSG((msg + "in SLMesh::init: " + _name).c_str());

    if (N.empty()) calcNormals();

    // Set default materials if no materials are assigned
    // If colors are available use diffuse color attribute shader
    // otherwise use the default gray material
    if (!mat())
    {
        if (!C.empty())
            mat(SLMaterialDefaultColorAttribute::instance());
        else
            mat(SLMaterialDefaultGray::instance());
    }

    // build tangents for bump mapping
    if (mat()->needsTangents() && !UV[0].empty() && T.empty())
        calcTangents();

    _isSelected = false;
}
//-----------------------------------------------------------------------------
//! Simplified drawing method for shadow map creation
/*! This is used from within SLShadowMap::drawNodesIntoDepthBufferRec
 */
void SLMesh::drawIntoDepthBuffer(SLSceneView* sv,
                                 SLNode*      node,
                                 SLMaterial*  depthMat)
{
    SLGLState* stateGL = SLGLState::instance();

    // Check data
    SLstring msg;
    if (P.empty())
        msg = "No vertex positions (P)\n";

    if (_primitive == PT_points && I16.empty() && I32.empty())
        msg += "No vertex indices (I16 or I32)\n";

    if (msg.length() > 0)
    {
        SL_WARN_MSG((msg + "in SLMesh::draw: " + _name).c_str());
        return;
    }

    // Return if hidden
    if (node->levelForSM() == 0 &&
        (node->drawBit(SL_DB_HIDDEN) || _primitive == PT_points))
        return;

    if (!_vao.vaoID())
        generateVAO(_vao);

    // Now use the depth material
    SLGLProgram* sp = depthMat->program();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mMatrix", 1, (SLfloat*)&stateGL->modelMatrix);
    sp->uniformMatrix4fv("u_vMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
    sp->uniformMatrix4fv("u_pMatrix", 1, (SLfloat*)&stateGL->projectionMatrix);

    _vao.drawElementsAs(PT_triangles);
}
//-----------------------------------------------------------------------------
/*!
SLMesh::draw does the OpenGL rendering of the mesh. The GL_TRIANGLES primitives
are rendered normally with the vertex position vector P, the normal vector N,
the vector UV1 and the index vector I16 or I32. GL_LINES & GL_POINTS don't have
normals and tex.coords. GL_POINTS don't have indexes (I16,I32) and are rendered
with glDrawArrays instead glDrawElements.
Optionally you can draw the normals and/or the uniform grid voxels.
<p> The method performs the following steps:</p>
<p>
1) Apply the drawing bits<br>
2) Generate Vertex Array Object once<br>
3) Apply the uniform variables to the shader<br>
3a) Activate a shader program if it is not yet in use and apply all its material parameters.<br>
3b) Pass the standard matrices to the shader program.<br>
4) Finally do the draw call by calling SLGLVertexArray::drawElementsAs<br>
5) Draw optional normals & tangents<br>
6) Draw optional acceleration structure<br>
7) Draw selected mesh with points<br>
</p>
Please view also the full process of rendering <a href="md_on_paint.html"><b>one frame</b></a>
*/
void SLMesh::draw(SLSceneView* sv, SLNode* node)
{
    SLGLState* stateGL = SLGLState::instance();

    // Check data
    SLstring msg;
    if (P.empty())
        msg = "No vertex positions (P)\n";
    if (_primitive != PT_points && I16.empty() && I32.empty())
        msg += "No vertex indices (I16 or I32)\n";
    if (msg.length() > 0)
    {
        SL_WARN_MSG((msg + "in SLMesh::draw: " + _name).c_str());
        return;
    }

    ////////////////////////
    // 1) Apply Drawing Bits
    ////////////////////////

    // Return if hidden
    if (sv->drawBit(SL_DB_HIDDEN) || node->drawBit(SL_DB_HIDDEN))
        return;

    SLGLPrimitiveType primitiveType = _primitive;

    // Set polygon mode
    if ((sv->drawBit(SL_DB_MESHWIRED) || node->drawBit(SL_DB_MESHWIRED)) &&
        typeid(*node) != typeid(SLSkybox))
    {
#ifdef SL_GLES
        primitiveType = PT_lineLoop; // There is no polygon line or point mode on ES2!
#else
        stateGL->polygonLine(true);
#endif
    }
    else
        stateGL->polygonLine(false);

    // Set face culling
    bool noFaceCulling = sv->drawBit(SL_DB_CULLOFF) || node->drawBit(SL_DB_CULLOFF);
    stateGL->cullFace(!noFaceCulling);

    // enable polygon offset if voxels are drawn to avoid stitching
    if (sv->drawBit(SL_DB_VOXELS) || node->drawBit(SL_DB_VOXELS))
        stateGL->polygonOffsetLine(true, -1.0f, -1.0f);

    ///////////////////////////////////////
    // 2) Generate Vertex Array Object once
    ///////////////////////////////////////

    if (!_vao.vaoID())
        generateVAO(_vao);

    /////////////////////////////
    // 3) Apply Uniform Variables
    /////////////////////////////

    // 3.a) Apply mesh material if exists & differs from current
    _mat->activate(sv->camera(), &sv->s()->lights());

    // 3.b) Pass the standard matrices to the shader program
    SLGLProgram* sp = _mat->program();
    sp->uniformMatrix4fv("u_mMatrix", 1, (SLfloat*)&stateGL->modelMatrix);
    sp->uniformMatrix4fv("u_vMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
    sp->uniformMatrix4fv("u_pMatrix", 1, (SLfloat*)&stateGL->projectionMatrix);

    SLint locTM = sp->getUniformLocation("u_tMatrix");
    if (locTM >= 0)
    {
        if (_mat->has3DTexture() && _mat->textures3d()[0]->autoCalcTM3D())
            calcTex3DMatrix(node);
        else
            stateGL->textureMatrix = _mat->textures(TT_diffuse)[0]->tm();
        sp->uniformMatrix4fv(locTM, 1, (SLfloat*)&stateGL->textureMatrix);
    }

    ///////////////////////////////
    // 4): Finally do the draw call
    ///////////////////////////////

    if ((sv->drawBit(SL_DB_ONLYEDGES) || node->drawBit(SL_DB_ONLYEDGES)) &&
        (!IE32.empty() || !IE16.empty()))
        _vao.drawEdges(_edgeColor, _edgeWidth);
    else
    {
        if (_primitive == PT_points)
            _vao.drawArrayAs(PT_points);
        else
        {
            _vao.drawElementsAs(primitiveType);

            if ((sv->drawBit(SL_DB_WITHEDGES) || node->drawBit(SL_DB_WITHEDGES)) &&
                (!IE32.empty() || !IE16.empty()))
            {
                stateGL->polygonOffsetLine(true, 1.0f, 1.0f);
                _vao.drawEdges(_edgeColor, _edgeWidth);
                stateGL->polygonOffsetLine(false);
            }
        }
    }

    //////////////////////////////////////
    // 5) Draw optional normals & tangents
    //////////////////////////////////////

    // All helper lines must be drawn without blending
    SLbool blended = stateGL->blend();
    if (blended) stateGL->blend(false);

    if (!N.empty() && (sv->drawBit(SL_DB_NORMALS) || node->drawBit(SL_DB_NORMALS)))
    {
        // Scale factor r 2% from scaled radius for normals & tangents
        // Build array between vertex and normal target point
        float    r = node->aabb()->radiusOS() * 0.02f;
        SLVVec3f V2;
        V2.resize(P.size() * 2);
        for (SLulong i = 0; i < P.size(); ++i)
        {
            V2[i << 1] = finalP((SLuint)i);
            V2[(i << 1) + 1].set(finalP((SLuint)i) + finalN((SLuint)i) * r);
        }

        // Create or update VAO for normals
        _vaoN.generateVertexPos(&V2);

        if (!T.empty())
        {
            for (SLulong i = 0; i < P.size(); ++i)
            {
                V2[(i << 1) + 1].set(finalP((SLuint)i).x + T[i].x * r,
                                     finalP((SLuint)i).y + T[i].y * r,
                                     finalP((SLuint)i).z + T[i].z * r);
            }

            // Create or update VAO for tangents
            _vaoT.generateVertexPos(&V2);
        }

        // Draw normals
        _vaoN.drawArrayAsColored(PT_lines, SLCol4f::BLUE);

        // Draw tangents if available
        if (!T.empty())
            _vaoT.drawArrayAsColored(PT_lines, SLCol4f::RED);
        if (blended)
            stateGL->blend(false);
    }
    else
    { // release buffer objects for normal & tangent rendering
        if (_vaoN.vaoID())
            _vaoN.deleteGL();
        if (_vaoT.vaoID())
            _vaoT.deleteGL();
    }

    //////////////////////////////////////////
    // 6) Draw optional acceleration structure
    //////////////////////////////////////////

    if (_accelStruct)
    {
        if (sv->drawBit(SL_DB_VOXELS) || node->drawBit(SL_DB_VOXELS))
        {
            _accelStruct->draw(sv);
            stateGL->polygonOffsetLine(false);
        }
        else
        { // Delete the visualization VBO if not rendered anymore
            _accelStruct->disposeBuffers();
        }
    }

    ////////////////////////////////////
    // 7: Draw selected mesh with points
    ////////////////////////////////////

    if (!node->drawBit(SL_DB_NOTSELECTABLE))
        handleRectangleSelection(sv, stateGL, node);

    if (blended)
        stateGL->blend(true);
}
//-----------------------------------------------------------------------------
//! Handles the rectangle section of mesh vertices (partial selection)
/*
 There are two different selection modes: Full or partial mesh selection.
 <br>
 The full selection is done by double-clicking a mesh. For more information
 see SLScene::selectNodeMesh.
 <br>
 Partial meshes can be selected by drawing a rectangle with CTRL-LMB. The
 selection rectangle is defined in the SLCamera::selectRect and gets set
 in SLCamera::onMouseDown, SLCamera::onMouseMove and SLCamera::onMouseUp.
 The partial selection in SLMesh::handleRectangleSelection. The selected
 vertices are stored in SLMesh::IS32. A mesh that is used in multiple nodes
 can only be partially selected from one node.
*/
void SLMesh::handleRectangleSelection(SLSceneView* sv,
                                      SLGLState*   stateGL,
                                      SLNode*      node)
{
    SLScene*  s   = sv->s();
    SLCamera* cam = sv->camera();

    // Single node and mesh is selected
    if (cam->selectRect().isEmpty() && cam->deselectRect().isEmpty())
    {
        if (node->isSelected() && _isSelected)
            drawSelectedVertices();
    }
    else // rect selection or deselection is going on
    {
        // Build full viewport-modelview-projection transform matrix
        SLMat4f mvp(stateGL->projectionMatrix * stateGL->viewMatrix * node->updateAndGetWM());
        SLMat4f v;
        SLRecti vp = sv->viewportRect();
        v.viewport((SLfloat)vp.x,
                   (SLfloat)vp.y,
                   (SLfloat)vp.width,
                   (SLfloat)vp.height);
        SLMat4f     v_mvp = v * mvp;
        set<SLuint> tempIselected;        // Temp. vector for selected vertex indices

        if (!cam->selectRect().isEmpty()) // Do rectangle Selection
        {
            // Select by transform all vertices and add the ones in the rect to tempInRect
            set<SLuint> tempInRect;
            for (SLulong i = 0; i < P.size(); ++i)
            {
                SLVec3f p = v_mvp * P[i];
                if (cam->selectRect().contains(SLVec2f(p.x, p.y)))
                    tempInRect.insert((SLuint)i);
            }

            // Merge the rectangle selected by doing a set union operation
            set<SLuint> IS32set(IS32.begin(), IS32.end());
            std::set_union(IS32set.begin(),
                           IS32set.end(),
                           tempInRect.begin(),
                           tempInRect.end(),
                           inserter(tempIselected, tempIselected.begin()));
        }
        else if (!cam->deselectRect().isEmpty()) // Do rectangle Deselection
        {
            // Deselect by transform all vertices and add the ones in the rect to tempIdeselected
            set<SLuint> tempIdeselected; // Temp. vector for deselected vertex indices
            for (SLulong i = 0; i < P.size(); ++i)
            {
                SLVec3f p = v_mvp * P[i];
                if (cam->deselectRect().contains(SLVec2f(p.x, p.y)))
                    tempIdeselected.insert((SLuint)i);
            }

            // Remove the deselected ones by doing a set difference operation
            tempIselected.clear();
            set<SLuint> IS32set(IS32.begin(), IS32.end());
            std::set_difference(IS32set.begin(),
                                IS32set.end(),
                                tempIdeselected.begin(),
                                tempIdeselected.end(),
                                inserter(tempIselected, tempIselected.begin()));
        }

        // Flag node and mesh for the first time a mesh gets rectangle selected.
        if (!tempIselected.empty() && !node->isSelected() && !_isSelected)
        {
            node->isSelected(true);
            s->selectedNodes().push_back(node);
            _isSelected = true;
            s->selectedMeshes().push_back(this);
            drawSelectedVertices();
        }

        // Do not rect-select if the mesh is not selected because it got
        // selected within another node.
        if (node->isSelected() && _isSelected)
        {
            if (!tempIselected.empty())
                IS32.assign(tempIselected.begin(), tempIselected.end());

            if (!IS32.empty())
                drawSelectedVertices();
        }
    }
}
//-----------------------------------------------------------------------------
/*! Clears the partial selection but not the flag SLMesh::_isSelected
 See also SLMesh::handleRectangleSelection.
 */
void SLMesh::deselectPartialSelection()
{
    _vaoS.clearAttribs();
    IS32.clear();
}
//-----------------------------------------------------------------------------
/*! If the entire mesh is selected all points will be drawn with an the vertex
 array only without indices. If a subset is selected we use the extra index
 array IS32. See also SLMesh::handleRectangleSelection.
 */
void SLMesh::drawSelectedVertices()

{
    SLGLState* stateGL = SLGLState::instance();
    stateGL->polygonOffsetPoint(true);
    stateGL->depthMask(false);
    stateGL->depthTest(false);

    if (IS32.empty())
    {
        // Draw all
        _vaoS.generateVertexPos(_finalP);
        _vaoS.drawArrayAsColored(PT_points, SLCol4f::YELLOW, 2);
    }
    else
    {
        // Draw only selected
        _vaoS.clearAttribs();
        _vaoS.setIndices(&IS32);
        _vaoS.generateVertexPos(_finalP);
        _vaoS.drawElementAsColored(PT_points, SLCol4f::YELLOW, 2);
    }

    stateGL->polygonLine(false);
    stateGL->polygonOffsetPoint(false);
    stateGL->depthMask(true);
    stateGL->depthTest(true);
}
//-----------------------------------------------------------------------------
//! Generate the Vertex Array Object for a specific shader program
void SLMesh::generateVAO(SLGLVertexArray& vao)
{
    PROFILE_FUNCTION();

    vao.setAttrib(AT_position, AT_position, _finalP);
    if (!N.empty()) vao.setAttrib(AT_normal, AT_normal, _finalN);
    if (!UV[0].empty()) vao.setAttrib(AT_uv1, AT_uv1, &UV[0]);
    if (!UV[1].empty()) vao.setAttrib(AT_uv2, AT_uv2, &UV[1]);
    if (!C.empty()) vao.setAttrib(AT_color, AT_color, &C);
    if (!T.empty()) vao.setAttrib(AT_tangent, AT_tangent, &T);

    if (!I16.empty() || !I32.empty())
    {
        // for triangle meshes compute hard edges and store indices behind the ones of the triangles
        if (_primitive == PT_triangles)
        {
            IE16.clear();
            IE32.clear();
            computeHardEdgesIndices(_edgeAngleDEG, _vertexPosEpsilon);
            if (!I16.empty()) vao.setIndices(&I16, &IE16);
            if (!I32.empty()) vao.setIndices(&I32, &IE32);
        }
        else
        {
            // for points there are no indices at all
            if (!I16.empty()) vao.setIndices(&I16);
            if (!I32.empty()) vao.setIndices(&I32);
        }
    }

    vao.generate((SLuint)P.size(),
                 !Ji.empty() ? BU_stream : BU_static,
                 Ji.empty());
}
//-----------------------------------------------------------------------------
//! computes the hard edges and stores the vertex indexes separately
/*! Hard edges are edges between faces where there normals have an angle
 greater than angleDEG (by default 30°). If a mesh has only smooth edges such
 as a sphere there will be no hard edges. The indices of those hard edges are
 stored in the vector IE16 or IE32. For rendering these indices are appended
 behind the indices for the triangle drawing. This is because the index the
 same vertices of the same VAO. See SLMesh::generateVAO for the details.
 */
void SLMesh::computeHardEdgesIndices(float angleDEG,
                                     float epsilon)
{
    // dihedral angle considered to sharp
    float angleRAD = angleDEG * Utils::DEG2RAD;

    if (_primitive != PT_triangles)
        return;

    Eigen::MatrixXf V;     // Input vertices for igl
    Eigen::MatrixXi F;     // Input faces (=triangle) indices for igl
    Eigen::MatrixXf newV;  // new vertices after duplicate removal
    Eigen::MatrixXi newF;  // new face indices after duplicate removal
    Eigen::MatrixXi edges; // all edges
    Eigen::MatrixXi edgeMap;
    Eigen::MatrixXi uniqueEdges;
    Eigen::MatrixXf faceN; // face normals
    Eigen::VectorXi SVI;   // new to old index mapping
    Eigen::VectorXi SVJ;   // old to new index mapping

    vector<vector<int>> uE2E;

    // fill input matrices
    V.resize((Eigen::Index)_finalP->size(), 3);
    for (int i = 0; i < _finalP->size(); i++)
        V.row(i) << finalP(i).x, finalP(i).y, finalP(i).z;

    if (!I16.empty())
    {
        F.resize((Eigen::Index)I16.size() / 3, 3);
        for (int j = 0, i = 0; i < I16.size(); j++, i += 3)
            F.row(j) << I16[i], I16[i + 1], I16[i + 2];
    }
    if (!I32.empty())
    {
        F.resize((Eigen::Index)I32.size() / 3, 3);
        for (int j = 0, i = 0; i < I32.size(); j++, i += 3)
            F.row(j) << I32[i], I32[i + 1], I32[i + 2];
    }

    // extract sharp edges
    igl::remove_duplicate_vertices(V, F, epsilon, newV, SVI, SVJ, newF);
    igl::per_face_normals(newV, newF, faceN);
    igl::unique_edge_map(newF, edges, uniqueEdges, edgeMap, uE2E);

    for (int u = 0; u < uE2E.size(); u++)
    {
        bool sharp = false;
        if (uE2E[u].size() == 1) // edges at the border (with only one triangle)
            sharp = true;

        // if more than one edge is passing here, compute dihedral angles
        for (int i = 0; i < uE2E[u].size(); i++)
        {
            for (int j = i + 1; j < uE2E[u].size(); j++)
            {
                // E[faceId + |F| * c] opposite of vertex F[faceId][c]
                // ei = fi + |F| * c -> ei % |F| = fi % |F| = fi; fi is the face adjacent to ei
                // ej = fj + |F| * c -> ej % |F| = fj % |F| = fj; fj is the face adjacent to ej
                const int                  ei  = uE2E[u][i]; // edge i
                const int                  fi  = ei % newF.rows();
                const int                  ej  = uE2E[u][j]; // edge j
                const int                  fj  = ej % newF.rows();
                Eigen::Matrix<float, 1, 3> ni  = faceN.row(fi);
                Eigen::Matrix<float, 1, 3> nj  = faceN.row(fj);
                Eigen::Matrix<float, 1, 3> ev  = (newV.row(edges(ei, 1)) - newV.row(edges(ei, 0))).normalized();
                float                      dij = Utils::PI - atan2((ni.cross(nj)).dot(ev), ni.dot(nj));
                sharp                          = std::abs(dij - Utils::PI) > angleRAD;
            }
        }

        if (sharp)
        {
            if (!I16.empty())
            {
                IE16.push_back(SVI[uniqueEdges(u, 0)]);
                IE16.push_back(SVI[uniqueEdges(u, 1)]);
            }
            else if (!I32.empty())
            {
                IE32.push_back(SVI[uniqueEdges(u, 0)]);
                IE32.push_back(SVI[uniqueEdges(u, 1)]);
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*!
SLMesh::hit does the ray-mesh intersection test. If no acceleration
structure is defined all triangles are tested in a brute force manner.
*/
SLbool SLMesh::hit(SLRay* ray, SLNode* node)
{
    // return true for point & line objects
    if (_primitive != PT_triangles)
    {
        ray->hitNode = node;
        ray->hitMesh = this;
        SLVec3f OC   = node->aabb()->centerWS() - ray->origin;
        ray->length  = OC.length();
        return true;
    }

    if (_accelStruct)
        return _accelStruct->intersect(ray, node);
    else
    { // intersect against all faces
        SLbool wasHit = false;

        for (SLuint t = 0; t < numI(); t += 3)
            if (hitTriangleOS(ray, node, t) && !wasHit)
                wasHit = true;

        return wasHit;
    }
}
//-----------------------------------------------------------------------------
/*!
SLMesh::updateStats updates the parent node statistics.
*/
void SLMesh::addStats(SLNodeStats& stats)
{
    stats.numBytes += sizeof(SLMesh);
    if (!P.empty()) stats.numBytes += SL_sizeOfVector(P);
    if (!N.empty()) stats.numBytes += SL_sizeOfVector(N);
    if (!UV[0].empty()) stats.numBytes += SL_sizeOfVector(UV[0]);
    if (!UV[1].empty()) stats.numBytes += SL_sizeOfVector(UV[1]);
    if (!C.empty()) stats.numBytes += SL_sizeOfVector(C);
    if (!T.empty()) stats.numBytes += SL_sizeOfVector(T);
    if (!Ji.empty()) stats.numBytes += SL_sizeOfVector(Ji);
    if (!Jw.empty()) stats.numBytes += SL_sizeOfVector(Jw);

    if (!I16.empty())
        stats.numBytes += (SLuint)(I16.size() * sizeof(SLushort));
    else
        stats.numBytes += (SLuint)(I32.size() * sizeof(SLuint));

    stats.numMeshes++;
    if (_primitive == PT_triangles) stats.numTriangles += numI() / 3;
    if (_primitive == PT_lines) stats.numLines += numI() / 2;

    if (_accelStruct)
        _accelStruct->updateStats(stats);
}
//-----------------------------------------------------------------------------
/*!
SLMesh::calcMinMax calculates the axis alligned minimum and maximum point
*/
void SLMesh::calcMinMax()
{
    // init min & max points
    minP.set(FLT_MAX, FLT_MAX, FLT_MAX);
    maxP.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // calc min and max point of all vertices
    for (SLulong i = 0; i < P.size(); ++i)
    {
        if (finalP((SLuint)i).x < minP.x) minP.x = finalP((SLuint)i).x;
        if (finalP((SLuint)i).x > maxP.x) maxP.x = finalP((SLuint)i).x;
        if (finalP((SLuint)i).y < minP.y) minP.y = finalP((SLuint)i).y;
        if (finalP((SLuint)i).y > maxP.y) maxP.y = finalP((SLuint)i).y;
        if (finalP((SLuint)i).z < minP.z) minP.z = finalP((SLuint)i).z;
        if (finalP((SLuint)i).z > maxP.z) maxP.z = finalP((SLuint)i).z;
    }
}
//-----------------------------------------------------------------------------
/*!
SLMesh::calcCenterRad calculates the center and the radius of an almost minimal
bounding sphere. Code by Jack Ritter from Graphic Gems.
*/
void SLMesh::calcCenterRad(SLVec3f& center, SLfloat& radius)
{
    SLulong i;
    SLfloat dx, dy, dz;
    SLfloat radius2, xspan, yspan, zspan, maxspan;
    SLfloat old_to_p, old_to_p_sq, old_to_new;
    SLVec3f xmin, xmax, ymin, ymax, zmin, zmax, dia1, dia2;

    // FIRST PASS: find 6 minima/maxima points
    xmin.x = ymin.y = zmin.z = FLT_MAX;
    xmax.x = ymax.y = zmax.z = -FLT_MAX;

    for (i = 0; i < P.size(); ++i)
    {
        if (P[i].x < xmin.x)
            xmin = P[i];
        else if (P[i].x > xmax.x)
            xmax = P[i];
        if (P[i].y < ymin.y)
            ymin = P[i];
        else if (P[i].y > ymax.y)
            ymax = P[i];
        if (P[i].z < zmin.z)
            zmin = P[i];
        else if (P[i].z > zmax.z)
            zmax = P[i];
    }

    // Set xspan = distance between the 2 points xmin & xmax (squared)
    dx    = xmax.x - xmin.x;
    dy    = xmax.y - xmin.y;
    dz    = xmax.z - xmin.z;
    xspan = dx * dx + dy * dy + dz * dz;

    // Same for y & z spans
    dx    = ymax.x - ymin.x;
    dy    = ymax.y - ymin.y;
    dz    = ymax.z - ymin.z;
    yspan = dx * dx + dy * dy + dz * dz;

    dx    = zmax.x - zmin.x;
    dy    = zmax.y - zmin.y;
    dz    = zmax.z - zmin.z;
    zspan = dx * dx + dy * dy + dz * dz;

    // Set points dia1 & dia2 to the maximally separated pair
    dia1    = xmin;
    dia2    = xmax; // assume xspan biggest
    maxspan = xspan;
    if (yspan > maxspan)
    {
        maxspan = yspan;
        dia1    = ymin;
        dia2    = ymax;
    }
    if (zspan > maxspan)
    {
        dia1 = zmin;
        dia2 = zmax;
    }

    // dia1,dia2 is a diameter of initial sphere
    // calc initial center
    center.x = (dia1.x + dia2.x) * 0.5f;
    center.y = (dia1.y + dia2.y) * 0.5f;
    center.z = (dia1.z + dia2.z) * 0.5f;

    // calculate initial radius*radius and radius
    dx      = dia2.x - center.x; // x component of radius vector
    dy      = dia2.y - center.y; // y component of radius vector
    dz      = dia2.z - center.z; // z component of radius vector
    radius2 = dx * dx + dy * dy + dz * dz;
    radius  = sqrt(radius2);

    // SECOND PASS: increment current sphere
    for (i = 0; i < P.size(); ++i)
    {
        dx          = P[i].x - center.x;
        dy          = P[i].y - center.y;
        dz          = P[i].z - center.z;
        old_to_p_sq = dx * dx + dy * dy + dz * dz;

        if (old_to_p_sq > radius2) // do r**2 test first
        {
            // this point is outside of current sphere
            old_to_p = sqrt(old_to_p_sq);

            // calc radius of new sphere
            radius     = (radius + old_to_p) * 0.5f;
            radius2    = radius * radius; // for next r**2 compare
            old_to_new = old_to_p - radius;

            // calc center of new sphere
            center.x = (radius * center.x + old_to_new * P[i].x) / old_to_p;
            center.y = (radius * center.y + old_to_new * P[i].y) / old_to_p;
            center.z = (radius * center.z + old_to_new * P[i].z) / old_to_p;

            // Suppress if desired
            SL_LOG("\n New sphere: center,radius = %f %f %f   %f",
                   center.x,
                   center.y,
                   center.z,
                   radius);
        }
    }
}
//-----------------------------------------------------------------------------
/*!
SLMesh::buildAABB builds the passed axis-aligned bounding box in OS and updates
the min & max points in WS with the passed WM of the node.
*/
void SLMesh::buildAABB(SLAABBox& aabb, const SLMat4f& wmNode)
{
    // Update acceleration struct and calculate min max
    if (_skeleton)
    {
        minP = _skeleton->minOS();
        maxP = _skeleton->maxOS();
    }
    else
    {
        // For now, we just update the acceleration struct for non-skinned meshes
        // Building the entire voxelization of a mesh every frame is not feasible
        if (_accelStructIsOutOfDate)
            updateAccelStruct();
    }
    // Apply world matrix
    aabb.fromOStoWS(minP, maxP, wmNode);
}
//-----------------------------------------------------------------------------
/*! SLMesh::updateAccelStruct rebuilds the acceleration structure if the dirty
flag is set. This can happen for mesh animations.
*/
void SLMesh::updateAccelStruct()
{
    calcMinMax();

    // Add half a percent in each direction to avoid zero size dimensions
    SLVec3f distMinMax = maxP - minP;
    SLfloat addon      = distMinMax.length() * 0.005f;
    minP -= addon;
    maxP += addon;

    if (_primitive != PT_triangles)
        return;

    if (_accelStruct == nullptr)
        _accelStruct = new SLCompactGrid(this);

    if (_accelStruct && numI() > 15)
    {
        _accelStruct->build(minP, maxP);
        _accelStructIsOutOfDate = false;
    }
}
//-----------------------------------------------------------------------------
//! SLMesh::calcNormals recalculates vertex normals for triangle meshes.
/*! SLMesh::calcNormals recalculates the normals only from the vertices.
This algorithms doesn't know anything about smoothgroups. It just loops over
the triangle of the material faces and sums up the normal for each of its
vertices. Note that the face normals are not normalized. The cross product of
2 vectors is proportional to the area of the triangle. Like this the normal of
big triangles are more weighted than small triangles and we get a better normal
quality. At the end all vertex normals are normalized.
*/
void SLMesh::calcNormals()
{
    // Set vector for the normals & Zero out the normals vector
    N.clear();

    if (_primitive != PT_triangles)
        return;

    // Create vector and fill with zero vectors
    N.resize(P.size());
    std::fill(N.begin(), N.end(), SLVec3f::ZERO);

    if (!I16.empty())
    {
        // Loop over all triangles
        for (SLulong i = 0; i < I16.size(); i += 3)
        {
            // Calculate the face's normal
            SLVec3f e1, e2, n;

            // Calculate edges of triangle
            e1.sub(P[I16[i + 1]], P[I16[i + 2]]); // e1 = B - C
            e2.sub(P[I16[i + 1]], P[I16[i]]);     // e2 = B - A

            // Build normal with cross product but do NOT normalize it.
            n.cross(e1, e2); // n = e1 x e2

            // Add this normal to its vertices normals
            N[I16[i]] += n;
            N[I16[i + 1]] += n;
            N[I16[i + 2]] += n;
        }
    }
    else
    {
        for (SLulong i = 0; i < I32.size(); i += 3)
        {
            // Calculate the face's normal
            SLVec3f e1, e2, n;

            // Calculate edges of triangle
            e1.sub(P[I32[i + 1]], P[I32[i + 2]]); // e1 = B - C
            e2.sub(P[I32[i + 1]], P[I32[i]]);     // e2 = B - A

            // Build normal with cross product but do NOT normalize it.
            n.cross(e1, e2); // n = e1 x e2

            // Add this normal to its vertices normals
            N[I32[i]] += n;
            N[I32[i + 1]] += n;
            N[I32[i + 2]] += n;
        }
    }

    // normalize vertex normals
    for (SLulong i = 0; i < P.size(); ++i)
        N[i].normalize();
}
//-----------------------------------------------------------------------------
//! SLMesh::calcTangents computes the tangents per vertex for triangle meshes.
/*! SLMesh::calcTangents computes the tangent and bi-tangent per vertex used
for GLSL normal map bump mapping. The code and mathematical derivation is in
detail explained in: http://www.terathon.com/code/tangent.html
*/
void SLMesh::calcTangents()
{
    if (!P.empty() &&
        !N.empty() && !UV[0].empty() &&
        (!I16.empty() || !I32.empty()))
    {
        // Delete old tangents
        T.clear();

        if (_primitive != PT_triangles)
            return;

        // allocate tangents
        T.resize(P.size());

        // allocate temp arrays for tangents
        SLVVec3f T1;
        T1.resize(P.size());
        fill(T1.begin(), T1.end(), SLVec3f::ZERO);
        SLVVec3f T2;
        T2.resize(P.size());
        fill(T2.begin(), T2.end(), SLVec3f::ZERO);

        SLuint iVA, iVB, iVC;
        SLuint numT = numI() / 3; // NO. of triangles

        for (SLuint t = 0; t < numT; ++t)
        {
            SLuint i = t * 3; // vertex index

            // Get the 3 vertex indices
            if (!I16.empty())
            {
                iVA = I16[i];
                iVB = I16[i + 1];
                iVC = I16[i + 2];
            }
            else
            {
                iVA = I32[i];
                iVB = I32[i + 1];
                iVC = I32[i + 2];
            }

            float x1 = P[iVB].x - P[iVA].x;
            float x2 = P[iVC].x - P[iVA].x;
            float y1 = P[iVB].y - P[iVA].y;
            float y2 = P[iVC].y - P[iVA].y;
            float z1 = P[iVB].z - P[iVA].z;
            float z2 = P[iVC].z - P[iVA].z;

            float s1 = UV[0][iVB].x - UV[0][iVA].x;
            float s2 = UV[0][iVC].x - UV[0][iVA].x;
            float t1 = UV[0][iVB].y - UV[0][iVA].y;
            float t2 = UV[0][iVC].y - UV[0][iVA].y;

            float   r = 1.0F / (s1 * t2 - s2 * t1);
            SLVec3f sdir((t2 * x1 - t1 * x2) * r,
                         (t2 * y1 - t1 * y2) * r,
                         (t2 * z1 - t1 * z2) * r);
            SLVec3f tdir((s1 * x2 - s2 * x1) * r,
                         (s1 * y2 - s2 * y1) * r,
                         (s1 * z2 - s2 * z1) * r);

            T1[iVA] += sdir;
            T1[iVB] += sdir;
            T1[iVC] += sdir;

            T2[iVA] += tdir;
            T2[iVB] += tdir;
            T2[iVC] += tdir;
        }

        for (SLulong i = 0; i < P.size(); ++i)
        {
            // Gram-Schmidt orthogonalization
            T[i] = T1[i] - N[i] * N[i].dot(T1[i]);
            T[i].normalize();

            // Calculate temp. bi-tangent and store its handedness in T.w
            SLVec3f bitangent;
            bitangent.cross(N[i], T1[i]);
            T[i].w = (bitangent.dot(T2[i]) < 0.0f) ? -1.0f : 1.0f;
        }
    }
}
//-----------------------------------------------------------------------------
/* Calculate the texture matrix for 3D texture mapping from the AABB so that
the texture volume surrounds the AABB centrally.
*/
void SLMesh::calcTex3DMatrix(SLNode* node)
{
    SLVec3f max = node->aabb()->maxOS();
    SLVec3f ctr = node->aabb()->centerOS();
    SLVec3f ext = max - ctr;

    // determine the scale factor s from the max. AABB extension
    SLint   dim;
    SLfloat maxExt = ext.maxXYZ(dim);
    SLfloat s      = 1.0f / (2.0f * maxExt);

    // scale and translate the texture matrix
    SLGLState* stateGL = SLGLState::instance();
    stateGL->textureMatrix.identity();
    stateGL->textureMatrix.scale(s);
    stateGL->textureMatrix.translate(-ctr);
    stateGL->textureMatrix.translate(-ext.comp[dim],
                                     -ext.comp[dim],
                                     -ext.comp[dim]);
}
//-----------------------------------------------------------------------------
/*!
SLMesh::hitTriangleOS is the fast and minimum storage ray-triangle
intersection test by Tomas Moeller and Ben Trumbore (Journal of graphics
tools 2, 1997).
*/
SLbool SLMesh::hitTriangleOS(SLRay* ray, SLNode* node, SLuint iT)
{
    assert(ray && "ray pointer is null");
    assert(node && "node pointer is null");
    assert(_mat && "material pointer is null");

    ++SLRay::tests;

    if (_primitive != PT_triangles)
        return false;

    // prevent self-intersection of triangle
    if (ray->srcMesh == this && ray->srcTriangle == (SLint)iT)
        return false;

    SLVec3f cornerA, cornerB, cornerC;
    SLVec3f e1, e2; // edge 1 and 2
    SLVec3f AO, K, Q;

    // get the corner vertices
    if (!I16.empty())
    {
        cornerA = finalP(I16[iT]);
        cornerB = finalP(I16[iT + 1]);
        cornerC = finalP(I16[iT + 2]);
    }
    else
    {
        cornerA = finalP(I32[iT]);
        cornerB = finalP(I32[iT + 1]);
        cornerC = finalP(I32[iT + 2]);
    }

    // find vectors for two edges sharing the triangle vertex A
    e1.sub(cornerB, cornerA);
    e2.sub(cornerC, cornerA);

    // begin calculating determinant - also used to calculate U parameter
    K.cross(ray->dirOS, e2);

    // if determinant is near zero, ray lies in plane of triangle
    const SLfloat det = e1.dot(K);

    SLfloat inv_det, t, u, v;

    // if ray is outside do test with face culling
    if (ray->isOutside && _isVolume)
    { // check only front side triangles

        // exit if ray is from behind or parallel
        if (det < FLT_EPSILON) return false;

        // calculate distance from corner A to ray origin
        AO.sub(ray->originOS, cornerA);

        // Calculate barycentric coordinates: u>0 && v>0 && u+v<=1
        u = AO.dot(K);
        if (u < 0.0f || u > det) return false;

        // prepare to test v parameter
        Q.cross(AO, e1);

        // calculate v parameter and test bounds
        v = Q.dot(ray->dirOS);
        if (v < 0.0f || u + v > det) return false;

        // calculate intersection distance t
        inv_det = 1.0f / det;
        t       = e2.dot(Q) * inv_det;

        // if intersection is closer replace ray intersection parameters
        if (t > ray->length || t < 0.0f) return false;

        ray->length = t;

        // scale down u & v so that u+v<=1
        ray->hitU = u * inv_det;
        ray->hitV = v * inv_det;
    }
    else
    { // check front & backside triangles
        // exit if ray is parallel
        if (det < FLT_EPSILON && det > -FLT_EPSILON) return false;

        inv_det = 1.0f / det;

        // calculate distance from corner A to ray origin
        AO.sub(ray->originOS, cornerA);

        // Calculate barycentric coordinates: u>0 && v>0 && u+v<=1
        u = AO.dot(K) * inv_det;
        if (u < 0.0f || u > 1.0f) return false;

        // prepare to test v parameter
        Q.cross(AO, e1);

        // calculate v parameter and test bounds
        v = Q.dot(ray->dirOS) * inv_det;
        if (v < 0.0f || u + v > 1.0f) return false;

        // calculate t, ray intersects triangle
        t = e2.dot(Q) * inv_det;

        // if intersection is closer replace ray intersection parameters
        if (t > ray->length || t < 0.0f) return false;

        ray->length = t;
        ray->hitU   = u;
        ray->hitV   = v;
    }

    ray->hitTriangle = (SLint)iT;
    ray->hitNode     = node;
    ray->hitMesh     = this;

    ++SLRay::intersections;

    return true;
}
//-----------------------------------------------------------------------------
/*!
SLMesh::preShade calculates the rest of the intersection information
after the final hit point is determined. Should be called just before the
shading when the final intersection point of the closest triangle was found.
*/
void SLMesh::preShade(SLRay* ray)
{
    if (_primitive != PT_triangles)
        return;

    // Get the triangle indices
    SLuint iA, iB, iC;
    if (!I16.empty())
    {
        iA = I16[(SLushort)ray->hitTriangle];
        iB = I16[(SLushort)ray->hitTriangle + 1];
        iC = I16[(SLushort)ray->hitTriangle + 2];
    }
    else
    {
        iA = I32[(SLuint)ray->hitTriangle];
        iB = I32[(SLuint)ray->hitTriangle + 1];
        iC = I32[(SLuint)ray->hitTriangle + 2];
    }

    // calculate the hit point in world space
    ray->hitPoint.set(ray->origin + ray->length * ray->dir);

    // calculate the interpolated normal with vertex normals in object space
    ray->hitNormal.set(finalN(iA) * (1 - (ray->hitU + ray->hitV)) +
                       finalN(iB) * ray->hitU +
                       finalN(iC) * ray->hitV);

    // transform normal back to world space
    SLMat3f wmN(ray->hitNode->updateAndGetWM().mat3());
    ray->hitNormal.set(wmN * ray->hitNormal);

    // for shading the normal is expected to be unit length
    ray->hitNormal.normalize();

    // calculate interpolated texture coordinates
    SLVGLTexture& diffuseTex = ray->hitMesh->mat()->textures(TT_diffuse);

    if (!diffuseTex.empty() &&
        !diffuseTex[0]->images().empty() &&
        !UV[0].empty())
    {
        SLVec2f Tu(UV[0][iB] - UV[0][iA]);
        SLVec2f Tv(UV[0][iC] - UV[0][iA]);
        SLVec2f tc(UV[0][iA] + ray->hitU * Tu + ray->hitV * Tv);
        ray->hitTexColor.set(diffuseTex[0]->getTexelf(tc.x, tc.y));

        // bump mapping
        SLVGLTexture& bumpTex = ray->hitMesh->mat()->textures(TT_normal);
        if (!bumpTex.empty() && !bumpTex[0]->images().empty())
        {
            if (!T.empty())
            {
                // calculate the interpolated tangent with vertex tangent in object space
                SLVec4f hitT(T[iA] * (1 - (ray->hitU + ray->hitV)) +
                             T[iB] * ray->hitU +
                             T[iC] * ray->hitV);

                SLVec3f T3(hitT.x, hitT.y, hitT.z);         // tangent with 3 components
                T3.set(wmN * T3);                           // transform tangent back to world space
                SLVec2f d   = bumpTex[0]->dudv(tc.x, tc.y); // slope of bump-map at tc
                SLVec3f Nrm = ray->hitNormal;               // unperturbated normal
                SLVec3f B(Nrm ^ T3);                        // bi-normal tangent B
                B *= T[iA].w;                               // correct handedness
                SLVec3f D(d.x * T3 + d.y * B);              // perturbation vector D
                Nrm += D;
                Nrm.normalize();
                ray->hitNormal.set(Nrm);
            }
        }

        // Get ambient occlusion
        SLVGLTexture& aoTex = ray->hitMesh->mat()->textures(TT_occlusion);
        if (!UV[1].empty())
        {
            SLVec2f Tu2(UV[1][iB] - UV[1][iA]);
            SLVec2f Tv2(UV[1][iC] - UV[1][iA]);
            SLVec2f tc2(UV[1][iA] + ray->hitU * Tu2 + ray->hitV * Tv2);

            if (!aoTex.empty())
                ray->hitAO = aoTex[0]->getTexelf(tc2.x, tc2.y).r;
        }
    }

    // calculate interpolated color for meshes with color attributes
    if (!ray->hitMesh->C.empty())
    {
        SLCol4f CA = ray->hitMesh->C[iA];
        SLCol4f CB = ray->hitMesh->C[iB];
        SLCol4f CC = ray->hitMesh->C[iC];
        ray->hitTexColor.set(CA * (1 - (ray->hitU + ray->hitV)) +
                             CB * ray->hitU +
                             CC * ray->hitV);
    }
}
//-----------------------------------------------------------------------------
//! Transforms the vertex positions and normals with by joint weights
/*! If the mesh is used for skinned skeleton animation this method transforms
each vertex and normal by max. four joints of the skeleton. Each joint has
a weight and an index. After the transform the VBO have to be updated.
This skinning process can also be done (a lot faster) on the GPU.
This software skinning is also needed for ray or path tracing.
*/
void SLMesh::transformSkin(const std::function<void(SLMesh*)>& cbInformNodes)
{
    // create the secondary buffers for P and N once
    if (skinnedP.empty())
    {
        skinnedP.resize(P.size());
        for (SLulong i = 0; i < P.size(); ++i)
            skinnedP[i] = P[i];
    }
    if (skinnedN.empty() && !N.empty())
    {
        skinnedN.resize(P.size());
        for (SLulong i = 0; i < P.size(); ++i)
            skinnedN[i] = N[i];
    }

    // Create array for joint matrices once
    if (_jointMatrices.empty())
    {
        _jointMatrices.clear();
        _jointMatrices.resize((SLuint)_skeleton->numJoints());
    }

    // update the joint matrix array
    _skeleton->getJointMatrices(_jointMatrices);

    // notify Parent Nodes to update AABB
    cbInformNodes(this);

    // temporarily set finalP and finalN
    _finalP = &skinnedP;
    _finalN = &skinnedN;

    // flag acceleration structure to be rebuilt
    _accelStructIsOutOfDate = true;

    // iterate over all vertices and write to new buffers
    for (SLulong i = 0; i < P.size(); ++i)
    {
        skinnedP[i] = SLVec3f::ZERO;
        if (!N.empty()) skinnedN[i] = SLVec3f::ZERO;

        // accumulate final normal and positions
        for (SLulong j = 0; j < Ji[i].size(); ++j)
        {
            const SLMat4f& jm      = _jointMatrices[Ji[i][j]];
            SLVec4f        tempPos = SLVec4f(jm * P[i]);
            skinnedP[i].x += tempPos.x * Jw[i][j];
            skinnedP[i].y += tempPos.y * Jw[i][j];
            skinnedP[i].z += tempPos.z * Jw[i][j];

            if (!N.empty())
            {
                // Build the 3x3 submatrix in GLSL 110 (= mat3 jt3 = mat3(jt))
                // for the normal transform that is the normally the inverse transpose.
                // The inverse transpose can be ignored as long as we only have
                // rotation and uniform scaling in the 3x3 submatrix.
                SLMat3f jnm = jm.mat3();
                skinnedN[i] += jnm * N[i] * Jw[i][j];
            }
        }
    }

    // update or create buffers
    if (_vao.vaoID())
    {
        _vao.updateAttrib(AT_position, _finalP);
        if (!N.empty()) _vao.updateAttrib(AT_normal, _finalN);
    }
}
//-----------------------------------------------------------------------------
#ifdef SL_HAS_OPTIX
unsigned int SLMesh::meshIndex = 0;
//-----------------------------------------------------------------------------
void SLMesh::allocAndUploadData()
{
    _vertexBuffer.alloc_and_upload(P);

    _normalBuffer.alloc_and_upload(N);

    if (!UV[0].empty())
        _textureBuffer.alloc_and_upload(UV[0]);

    if (!UV[1].empty())
        _textureBuffer.alloc_and_upload(UV[1]);

    if (!I16.empty())
        _indexShortBuffer.alloc_and_upload(I16);
    else
        _indexIntBuffer.alloc_and_upload(I32);
}
//-----------------------------------------------------------------------------
void SLMesh::uploadData()
{
    _vertexBuffer.upload(*_finalP);
    _normalBuffer.upload(*_finalN);
}
//-----------------------------------------------------------------------------
void SLMesh::createMeshAccelerationStructure()
{
    if (!_vertexBuffer.isAllocated())
    {
        allocAndUploadData();
    }

    // Build triangle GAS
    uint32_t _buildInput_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

    _buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    if (!I16.empty())
    {
        _buildInput.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
        _buildInput.triangleArray.numIndexTriplets = (SLuint)(I16.size() / 3);
        _buildInput.triangleArray.indexBuffer      = _indexShortBuffer.devicePointer();
    }
    else
    {
        _buildInput.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        _buildInput.triangleArray.numIndexTriplets = (SLuint)(I32.size() / 3);
        _buildInput.triangleArray.indexBuffer      = _indexIntBuffer.devicePointer();
    }
    _buildInput.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    _buildInput.triangleArray.vertexBuffers               = _vertexBuffer.devicePointerPointer();
    _buildInput.triangleArray.numVertices                 = (SLuint)P.size();
    _buildInput.triangleArray.flags                       = _buildInput_flags;
    _buildInput.triangleArray.numSbtRecords               = 1;
    _buildInput.triangleArray.sbtIndexOffsetBuffer        = 0;
    _buildInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
    _buildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    _sbtIndex = RAY_TYPE_COUNT * meshIndex++;

    buildAccelerationStructure();
}
//-----------------------------------------------------------------------------
void SLMesh::updateMeshAccelerationStructure()
{
    if (!_accelStructIsOutOfDate)
        return;

    uploadData();

    // Build triangle GAS
    uint32_t _buildInput_flags[1]   = {OPTIX_GEOMETRY_FLAG_NONE};
    _buildInput.triangleArray.flags = _buildInput_flags;

    updateAccelerationStructure();
}
//-----------------------------------------------------------------------------
ortHitData SLMesh::createHitData()
{
    ortHitData hitData = {};

    hitData.sbtIndex = 0;
    hitData.normals  = reinterpret_cast<float3*>(_normalBuffer.devicePointer());
    hitData.indices  = reinterpret_cast<short3*>(_indexShortBuffer.devicePointer());
    hitData.texCords = reinterpret_cast<float2*>(_textureBuffer.devicePointer());
    if (mat()->numTextures())
    {
        hitData.textureObject = mat()->textures(TT_diffuse)[0]->getCudaTextureObject();
    }
    hitData.material.kn                = mat()->kn();
    hitData.material.kt                = mat()->kt();
    hitData.material.kr                = mat()->kr();
    hitData.material.shininess         = mat()->shininess();
    hitData.material.ambient_color     = make_float4(mat()->ambient());
    hitData.material.specular_color    = make_float4(mat()->specular());
    hitData.material.transmissiv_color = make_float4(mat()->transmissive());
    hitData.material.diffuse_color     = make_float4(mat()->diffuse());
    hitData.material.emissive_color    = make_float4(mat()->emissive());

    return hitData;
}
#endif
//-----------------------------------------------------------------------------
