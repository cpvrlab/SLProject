//#############################################################################
//  File:      SLMesh.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLCompactGrid.h>
#include <SLNode.h>
#include <SLRay.h>
#include <SLRaytracer.h>
#include <SLSceneView.h>
#include <SLSkybox.h>

//-----------------------------------------------------------------------------
/*! 
The constructor initializes everything to 0 and adds the instance to the vector
SLScene::_meshes. All meshes are held globally in this vector and are deallocated
in SLScene::unInit().
*/
SLMesh::SLMesh(const SLstring& name) : SLObject(name)
{
    _primitive = PT_triangles;
    mat(nullptr);
    matOut(nullptr);
    _finalP = &P;
    _finalN = &N;
    minP.set(FLT_MAX, FLT_MAX, FLT_MAX);
    maxP.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    _skeleton             = nullptr;
    _isVolume             = true;    // is used for RT to decide inside/outside
    _accelStruct          = nullptr; // no initial acceleration structure
    _accelStructOutOfDate = true;

    // Add this mesh to the global resource vector for deallocation
    SLApplication::scene->meshes().push_back(this);
}
//-----------------------------------------------------------------------------
//! The destructor deletes everything by calling deleteData.
/*! All meshes are held globally in the vector SLScene::_meshes and are
deallocated when the scene is disposed in SLScene::unInit().
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
    Tc.clear();
    for (auto i : Ji)
        i.clear();
    Ji.clear();
    for (auto i : Jw)
        i.clear();
    Jw.clear();
    I16.clear();
    I32.clear();
    IS32.clear();

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
}
//-----------------------------------------------------------------------------
//! Deletes the rectangle selected vertices and the dependend triangles.
/*! The selection rectangle is defined in SLScene::selectRect and gets set and
 drawn in SLCamera::onMouseDown and SLCamera::onMouseMove. If the selectRect is
 not empty the SLScene::selectedNode is null. All vertices that are within the
 selectRect are listed in SLMesh::IS32. The selection evaluation is done during
 drawing in SLMesh::draw and is only valid for the current frame.
 All nodes that have selected vertice have their drawbit SL_DB_SELECTED set. */
void SLMesh::deleteSelected(SLNode* node)
{
    // Loop over all rectangle selected indexes in IS32
    for (SLulong i = 0; i < IS32.size(); ++i)
    {
        SLuint ixDel = IS32[i] - i;

        if (ixDel < P.size()) P.erase(P.begin() + ixDel);
        if (ixDel < N.size()) N.erase(N.begin() + ixDel);
        if (ixDel < C.size()) C.erase(C.begin() + ixDel);
        if (ixDel < T.size()) T.erase(T.begin() + ixDel);
        if (ixDel < Tc.size()) Tc.erase(Tc.begin() + ixDel);
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
    if (mat()->needsTangents() && !Tc.empty() && T.empty())
        calcTangents();

    // delete vertex array object so it gets regenerated
    _vao.deleteGL();

    // delete the selection indexes
    IS32.clear();

    // flag aabb and aceleration structure to be updated
    node->needAABBUpdate();
    _accelStructOutOfDate = true;
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
            SLuint ixDel = u - (unused - 1);

            if (ixDel < P.size()) P.erase(P.begin() + ixDel);
            if (ixDel < N.size()) N.erase(N.begin() + ixDel);
            if (ixDel < C.size()) C.erase(C.begin() + ixDel);
            if (ixDel < T.size()) T.erase(T.begin() + ixDel);
            if (ixDel < Tc.size()) Tc.erase(Tc.begin() + ixDel);
            if (ixDel < Ji.size()) Ji.erase(Ji.begin() + ixDel);
            if (ixDel < Jw.size()) Jw.erase(Jw.begin() + ixDel);

            // decrease the indexes smaller than the deleted on
            for (SLulong i = 0; i < I16.size(); ++i)
            {
                if (I16[i] > ixDel)
                    I16[i]--;
            }

            for (SLulong i = 0; i < I32.size(); ++i)
            {
                if (I32[i] > ixDel)
                    I32[i]--;
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

    // Set default materials if no materials are asigned
    // If colors are available use diffuse color attribute shader
    // otherwise use the default gray material
    if (!mat())
    {
        if (!C.empty())
            mat(SLMaterial::diffuseAttrib());
        else
            mat(SLMaterial::defaultGray());
    }

    // set transparent flag of the node if mesh contains alpha material
    if (!node->aabb()->hasAlpha() && mat()->hasAlpha())
        node->aabb()->hasAlpha(true);

    // build tangents for bump mapping
    if (mat()->needsTangents() && !Tc.empty() && T.empty())
        calcTangents();
}
//-----------------------------------------------------------------------------
/*! 
SLMesh::draw does the OpenGL rendering of the mesh. The GL_TRIANGLES primitives
are rendered normally with the vertex position vector P, the normal vector N,
the vector Tc and the index vector I16 or I32. GL_LINES & GL_POINTS don't have
normals and tex.coords. GL_POINTS don't have indexes (I16,I32) and are rendered
with glDrawArrays instead glDrawElements.
Optionally you can draw the normals and/or the uniform grid voxels.
<p> The method performs the following steps:</p>
<p>
1) Apply the drawing bits<br>
2) Apply the uniform variables to the shader<br>
2a) Activate a shader program if it is not yet in use and apply all its material parameters.<br>
2b) Pass the modelview and modelview-projection matrix to the shader.<br>
2c) If needed build and pass the inverse modelview and the normal matrix.<br>
2d) If the mesh has a skeleton and HW skinning is applied pass the joint matrices.<br>
3) Generate Vertex Array Object once<br>
4) Finally do the draw call<br>
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
    if ((sv->drawBit(SL_DB_WIREMESH) || node->drawBit(SL_DB_WIREMESH)) &&
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

    // check if texture exists
    //SLbool useTexture = Tc.size() && !sv->drawBit(SL_DB_TEXOFF) && !node->drawBit(SL_DB_TEXOFF);

    // enable polygon offset if voxels are drawn to avoid stitching
    if (sv->drawBit(SL_DB_VOXELS) || node->drawBit(SL_DB_VOXELS))
        stateGL->polygonOffset(true, 1.0f, 1.0f);

    /////////////////////////////
    // 2) Apply Uniform Variables
    /////////////////////////////

    // 2.a) Apply mesh material if exists & differs from current
    if (mat() != SLMaterial::current || SLMaterial::current->program() == nullptr)
        mat()->activate(*node->drawBits());

    // 2.b) Pass the matrices to the shader program
    SLGLProgram* sp = SLMaterial::current->program();
    sp->uniformMatrix4fv("u_mvMatrix", 1, (SLfloat*)&stateGL->modelViewMatrix);
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (const SLfloat*)stateGL->mvpMatrix());

    // 2.c) Build & pass inverse, normal & texture matrix only if needed
    SLint locIM = sp->getUniformLocation("u_invMvMatrix");
    SLint locNM = sp->getUniformLocation("u_nMatrix");
    SLint locTM = sp->getUniformLocation("u_tMatrix");

    if (locIM >= 0 && locNM >= 0)
    {
        stateGL->buildInverseAndNormalMatrix();
        sp->uniformMatrix4fv(locIM, 1, (const SLfloat*)stateGL->invModelViewMatrix());
        sp->uniformMatrix3fv(locNM, 1, (const SLfloat*)stateGL->normalMatrix());
    }
    else if (locIM >= 0)
    {
        stateGL->buildInverseMatrix();
        sp->uniformMatrix4fv(locIM, 1, (const SLfloat*)stateGL->invModelViewMatrix());
    }
    else if (locNM >= 0)
    {
        stateGL->buildNormalMatrix();
        sp->uniformMatrix3fv(locNM, 1, (const SLfloat*)stateGL->normalMatrix());
    }
    if (locTM >= 0)
    {
        if (_mat->has3DTexture() && _mat->textures()[0]->autoCalcTM3D())
            calcTex3DMatrix(node);
        else
            stateGL->textureMatrix = _mat->textures()[0]->tm();
        sp->uniformMatrix4fv(locTM, 1, (SLfloat*)&stateGL->textureMatrix);
    }

    ///////////////////////////////////////
    // 3) Generate Vertex Array Object once
    ///////////////////////////////////////

    if (!_vao.vaoID())
        generateVAO(sp);

    ///////////////////////////////
    // 4): Finally do the draw call
    ///////////////////////////////

    if (_primitive == PT_points)
        _vao.drawArrayAs(PT_points);
    else
        _vao.drawElementsAs(primitiveType);

    //////////////////////////////////////
    // 5) Draw optional normals & tangents
    //////////////////////////////////////

    // All helper lines must be drawn without blending
    SLbool blended = stateGL->blend();
    if (blended) stateGL->blend(false);

    if (!N.empty() && (sv->drawBit(SL_DB_NORMALS) || node->drawBit(SL_DB_NORMALS)))
    {
        // scale factor r 2% from scaled radius for normals & tangents
        // build array between vertex and normal target point
        float    r = node->aabb()->radiusOS() * 0.02f;
        SLVVec3f V2;
        V2.resize(P.size() * 2);
        for (SLulong i = 0; i < P.size(); ++i)
        {
            V2[i << 1] = finalP(i);
            V2[(i << 1) + 1].set(finalP(i) + finalN(i) * r);
        }

        // Create or update VAO for normals
        _vaoN.generateVertexPos(&V2);

        if (!T.empty())
        {
            for (SLulong i = 0; i < P.size(); ++i)
            {
                V2[(i << 1) + 1].set(finalP(i).x + T[i].x * r,
                                     finalP(i).y + T[i].y * r,
                                     finalP(i).z + T[i].z * r);
            }

            // Create or update VAO for tangents
            _vaoT.generateVertexPos(&V2);
        }

        _vaoN.drawArrayAsColored(PT_lines, SLCol4f::BLUE);
        if (!T.empty()) _vaoT.drawArrayAsColored(PT_lines, SLCol4f::RED);
        if (blended) stateGL->blend(false);
    }
    else
    { // release buffer objects for normal & tangent rendering
        if (_vaoN.vaoID()) _vaoN.deleteGL();
        if (_vaoT.vaoID()) _vaoT.deleteGL();
    }

    //////////////////////////////////////////
    // 6) Draw optional acceleration structure
    //////////////////////////////////////////

    if (_accelStruct)
    {
        if (sv->drawBit(SL_DB_VOXELS) || node->drawBit(SL_DB_VOXELS))
        {
            _accelStruct->draw(sv);
            stateGL->polygonOffset(false);
        }
        else
        { // Delete the visualization VBO if not rendered anymore
            _accelStruct->disposeBuffers();
        }
    }

    ////////////////////////////////////
    // 7: Draw selected mesh with points
    ////////////////////////////////////
    SLScene* s = SLApplication::scene;

    if (s->selectedNode() == node &&
        s->selectedMesh() == this)
    {
        stateGL->polygonOffset(true, 1.0f, 1.0f);
        stateGL->depthMask(false);
        stateGL->depthTest(false);
        _vaoS.generateVertexPos(_finalP);
        _vaoS.drawArrayAsColored(PT_points, SLCol4f::YELLOW, 2);
        stateGL->polygonLine(false);
        stateGL->polygonOffset(false);
        stateGL->depthMask(true);
        stateGL->depthTest(true);
    }
    else if (!s->selectedRect().isEmpty())
    {
        /* The selection rectangle is defined in SLScene::selectRect and gets set and
         drawn in SLCamera::onMouseDown and SLCamera::onMouseMove. If the selectRect is
         not empty the SLScene::selectedNode is null. All vertices that are within the
         selectRect are listed in SLMesh::IS32. The selection evaluation is done during
         drawing in SLMesh::draw and is only valid for the current frame.
         All nodes that have selected vertice have their drawbit SL_DB_SELECTED set. */

        // Build full viewport-modelview-projection transform
        SLMat4f mvp = *stateGL->mvpMatrix();
        SLMat4f v;
        SLRecti vp = sv->viewportRect();
        v.viewport((SLfloat)vp.x, (SLfloat)vp.y, (SLfloat)vp.width, (SLfloat)vp.height);
        SLMat4f v_mvp = v * mvp;
        IS32.clear();

        // Transform all verices and add the ones in the ROI to IS32
        for (SLulong i = 0; i < P.size(); ++i)
        {
            SLVec3f p = v_mvp * P[i];
            if (s->selectedRect().contains(SLVec2f(p.x, p.y)))
                IS32.push_back(i);
        }

        if (!IS32.empty())
        {
            stateGL->polygonOffset(true, 1.0f, 1.0f);
            stateGL->depthMask(false);
            stateGL->depthTest(false);
            _vaoS.clearAttribs();
            _vaoS.setIndices(&IS32);
            _vaoS.generateVertexPos(_finalP);
            _vaoS.drawElementAsColored(PT_points, SLCol4f::YELLOW, 2);
            stateGL->polygonLine(false);
            stateGL->polygonOffset(false);
            stateGL->depthMask(true);
            stateGL->depthTest(true);
            node->drawBits()->on(SL_DB_SELECTED);
        }
        else
            node->drawBits()->off(SL_DB_SELECTED);
    }
    else
    {
        if (_vaoS.vaoID())
        {
            _vaoS.clearAttribs();
            IS32.clear();
        }

        if (s->selectedNode() == nullptr && s->selectedRect().isEmpty())
            node->drawBits()->off(SL_DB_SELECTED);
    }

    if (blended) stateGL->blend(true);
}
//-----------------------------------------------------------------------------
//! Generate the Vertex Array Object for a specific shader program
void SLMesh::generateVAO(SLGLProgram* sp)
{
    _vao.setAttrib(AT_position, sp->getAttribLocation("a_position"), _finalP);
    if (!N.empty()) _vao.setAttrib(AT_normal, sp->getAttribLocation("a_normal"), _finalN);
    if (!Tc.empty()) _vao.setAttrib(AT_texCoord, sp->getAttribLocation("a_texCoord"), &Tc);
    if (!C.empty()) _vao.setAttrib(AT_color, sp->getAttribLocation("a_color"), &C);
    if (!T.empty()) _vao.setAttrib(AT_tangent, sp->getAttribLocation("a_tangent"), &T);
    if (!I16.empty()) _vao.setIndices(&I16);
    if (!I32.empty()) _vao.setIndices(&I32);

    _vao.generate((SLuint)P.size(), !Ji.empty() ? BU_stream : BU_static, Ji.empty());
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
    if (!Tc.empty()) stats.numBytes += SL_sizeOfVector(Tc);
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
        if (finalP(i).x < minP.x) minP.x = finalP(i).x;
        if (finalP(i).x > maxP.x) maxP.x = finalP(i).x;
        if (finalP(i).y < minP.y) minP.y = finalP(i).y;
        if (finalP(i).y > maxP.y) maxP.y = finalP(i).y;
        if (finalP(i).z < minP.z) minP.z = finalP(i).z;
        if (finalP(i).z > maxP.z) maxP.z = finalP(i).z;
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
    // update acceleration struct and calculate min max
    if (_skeleton)
    {
        minP = _skeleton->minOS();
        maxP = _skeleton->maxOS();
    }
    else
    { // for now we just update the acceleration struct for non skinned meshes
        // Building the entire voxelization of a mesh every frame is not feasible
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
    if (!_accelStructOutOfDate)
        return;

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
        _accelStructOutOfDate = false;
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
    if (!P.empty() && !N.empty() && !Tc.empty() && (!I16.empty() || !I32.empty()))
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
        SLuint numT = numI() / 3; //NO. of triangles

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

            float s1 = Tc[iVB].x - Tc[iVA].x;
            float s2 = Tc[iVC].x - Tc[iVA].x;
            float t1 = Tc[iVB].y - Tc[iVA].y;
            float t2 = Tc[iVC].y - Tc[iVA].y;

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

            // Calculate temp. bitangent and store its handedness in T.w
            SLVec3f bitangent;
            bitangent.cross(N[i], T1[i]);
            T[i].w = (bitangent.dot(T2[i]) < 0.0f) ? -1.0f : 1.0f;
        }
    }
}
//-----------------------------------------------------------------------------
/* Calculate the texture matrix for 3D texture mapping from the AABB so that
the texture volume surrounds the AABB centrically.
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

    SLVec3f A, B, C; // corners
    SLVec3f e1, e2;  // edge 1 and 2
    SLVec3f AO, K, Q;

    // get the corner vertices
    if (!I16.empty())
    {
        A = finalP(I16[iT]);
        B = finalP(I16[iT + 1]);
        C = finalP(I16[iT + 2]);
    }
    else
    {
        A = finalP(I32[iT]);
        B = finalP(I32[iT + 1]);
        C = finalP(I32[iT + 2]);
    }

    // find vectors for two edges sharing the triangle vertex A
    e1.sub(B, A);
    e2.sub(C, A);

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

        // calculate distance from A to ray origin
        AO.sub(ray->originOS, A);

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

        // calculate distance from A to ray origin
        AO.sub(ray->originOS, A);

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
    ray->hitNormal.set(ray->hitNode->updateAndGetWMN() * ray->hitNormal);

    // for shading the normal is expected to be unit length
    ray->hitNormal.normalize();

    // calculate interpolated texture coordinates
    SLVGLTexture& textures = ray->hitMesh->mat()->textures();
    if (!textures.empty() && !Tc.empty())
    {
        SLVec2f Tu(Tc[iB] - Tc[iA]);
        SLVec2f Tv(Tc[iC] - Tc[iA]);
        SLVec2f tc(Tc[iA] + ray->hitU * Tu + ray->hitV * Tv);
        ray->hitColor.set(textures[0]->getTexelf(tc.x, tc.y));

        // bump mapping
        if (textures.size() > 1)
        {
            if (!T.empty())
            {
                // calculate the interpolated tangent with vertex tangent in object space
                SLVec4f hitT(T[iA] * (1 - (ray->hitU + ray->hitV)) +
                             T[iB] * ray->hitU +
                             T[iC] * ray->hitV);

                SLVec3f T3(hitT.x, hitT.y, hitT.z);           // tangent with 3 components
                T3.set(ray->hitNode->updateAndGetWMN() * T3); // transform tangent back to world space
                SLVec2f d = textures[1]->dsdt(tc.x, tc.y);    // slope of bumpmap at tc
                SLVec3f N = ray->hitNormal;                   // unperturbated normal
                SLVec3f B(N ^ T3);                            // binormal tangent B
                B *= T[iA].w;                                 // correct handedness
                SLVec3f D(d.x * T3 + d.y * B);                // perturbation vector D
                N += D;
                N.normalize();
                ray->hitNormal.set(N);
            }
        }
    }

    // calculate interpolated color for meshes with color attributes
    if (!ray->hitMesh->C.empty())
    {
        SLCol4f CA = ray->hitMesh->C[iA];
        SLCol4f CB = ray->hitMesh->C[iB];
        SLCol4f CC = ray->hitMesh->C[iC];
        ray->hitColor.set(CA * (1 - (ray->hitU + ray->hitV)) +
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
void SLMesh::transformSkin()
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

    notifyParentNodesAABBUpdate();

    // temporarily set finalP and finalN
    _finalP = &skinnedP;
    _finalN = &skinnedN;

    // flag acceleration structure to be rebuilt
    _accelStructOutOfDate = true;

    // iterate over all vertices and write to new buffers
    for (SLulong i = 0; i < P.size(); ++i)
    {
        skinnedP[i] = SLVec3f::ZERO;
        if (!N.empty()) skinnedN[i] = SLVec3f::ZERO;

        // accumulate final normal and positions
        for (SLulong j = 0; j < Ji[i].size(); ++j)
        {
            const SLMat4f& jm      = _jointMatrices[Ji[i][j]];
            SLVec4f        tempPos = jm * P[i];
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
void SLMesh::notifyParentNodesAABBUpdate() const
{
    SLVNode nodes = SLApplication::scene->root3D()->findChildren(this);
    for (auto node : nodes)
        node->needAABBUpdate();
}
//-----------------------------------------------------------------------------
