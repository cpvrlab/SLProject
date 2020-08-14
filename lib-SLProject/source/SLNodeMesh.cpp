//#############################################################################
//  File:      SLNodeMesh.cpp
//  Author:    Marcus Hudritsch
//  Date:      August2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLNodeMesh.h>
#include <SLNode.h>
#include <SLMesh.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
/*!
*/
void SLNodeMesh::draw(SLSceneView* sv, SLMaterial* _mat)
{
    SLGLState* stateGL = SLGLState::instance();

    ///////////////////////////////////////
    // 1) Generate Vertex Array Object once
    ///////////////////////////////////////

    if (!_vao.vaoID())
    {
        // do expensive mesh access only once

        // Check data
        SLstring msg;
        if (_mesh->P.empty())
            msg = "No vertex positions (P)\n";

        if (_primitive != PT_points && _mesh->I16.empty() && _mesh->I32.empty())
            msg += "No vertex indices (I16 or I32)\n";
        if (msg.length() > 0)
        {
            SL_WARN_MSG((msg + "in SLMesh::draw: " + _mesh->name()).c_str());
            return;
        }

        _mesh->generateVAO(_vao);
        _primitive = _mesh->primitive();
    }



    ////////////////////////
    // 1) Apply Drawing Bits
    ////////////////////////

    // Return if hidden
    if (sv->drawBit(SL_DB_HIDDEN) || _node->drawBit(SL_DB_HIDDEN))
        return;


    // Set polygon mode
    if ((sv->drawBit(SL_DB_MESHWIRED) || _node->drawBit(SL_DB_MESHWIRED)) &&
        typeid(*_node) != typeid(SLSkybox))
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
    bool noFaceCulling = sv->drawBit(SL_DB_CULLOFF) || _node->drawBit(SL_DB_CULLOFF);
    stateGL->cullFace(!noFaceCulling);

    // check if texture exists
    //bool useTexture = Tc.size() && !sv->drawBit(SL_DB_TEXOFF) && !node->drawBit(SL_DB_TEXOFF);

    // enable polygon offset if voxels are drawn to avoid stitching
    if (sv->drawBit(SL_DB_VOXELS) || _node->drawBit(SL_DB_VOXELS))
        stateGL->polygonOffset(true, 1.0f, 1.0f);


    /////////////////////////////
    // 3) Apply Uniform Variables
    /////////////////////////////

    // We assume that the the material is activated
    //_mat->activate(*_node->drawBits(), sv->camera(), &sv->s()->lights());

    // 3.b) Pass the matrices to the shader program
    SLGLProgram* sp = _mat->program();
    sp->uniformMatrix4fv("u_mMatrix", 1, (SLfloat*)&_node->updateAndGetWM());
    sp->uniformMatrix4fv("u_mvMatrix", 1, (SLfloat*)&stateGL->modelViewMatrix);
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (const SLfloat*)stateGL->mvpMatrix());

    // 3.c) Build & pass inverse, normal & texture matrix only if needed
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
            _mesh->calcTex3DMatrix(_node);
        else
            stateGL->textureMatrix = _mat->textures()[0]->tm();
        sp->uniformMatrix4fv(locTM, 1, (SLfloat*)&stateGL->textureMatrix);
    }

    ///////////////////////////////
    // 4): Finally do the draw call
    ///////////////////////////////

    if (_primitive == PT_points)
        _vao.drawArrayAs(PT_points);
    else
        _vao.drawElementsAs(_primitive);

    /*
    //////////////////////////////////////
    // 5) Draw optional normals & tangents
    //////////////////////////////////////

    // All helper lines must be drawn without blending
    SLbool blended = stateGL->blend();
    if (blended)
        stateGL->blend(false);

    if (!N.empty() && (sv->drawBit(SL_DB_NORMALS) || _node->drawBit(SL_DB_NORMALS)))
    {
        // scale factor r 2% from scaled radius for normals & tangents
        // build array between vertex and normal target point
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

    if (!_node->drawBit(SL_DB_NOTSELECTABLE))
        handleRectangleSelection(sv, stateGL, _node);

    if (blended)
        stateGL->blend(true);

    */
}

//-----------------------------------------------------------------------------
