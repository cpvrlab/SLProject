//#############################################################################
//  File:      SLMesh.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLNode.h>
#include <SLMesh.h>
#include <SLRay.h>
#include <SLRaytracer.h>
#include <SLSceneView.h>
#include <SLCamera.h>
#include <SLUniformGrid.h>
#include <SLLightSphere.h>
#include <SLLightRect.h>
#include <SLSkeleton.h>
#include <SLGLProgram.h>

//-----------------------------------------------------------------------------
/*! 
The constructor initialises everything to 0 and adds the instance to the vector
SLScene::_meshes. All meshes are held globally in this vector and are deallocated
in SLScene::unInit().
*/
SLMesh::SLMesh(SLstring name) : SLObject(name)
{   
    _primitive = SL_TRIANGLES;
    cpuSkinningP = nullptr;
    cpuSkinningN = nullptr;
    P   = nullptr;
    N   = nullptr;
    C   = nullptr;
    T   = nullptr;
    Tc  = nullptr;
    Ji  = nullptr;
    Jw  = nullptr;
    I16 = nullptr;
    I32 = nullptr;
    mat = nullptr;
    _finalP = &P;
    _finalN = &N;
    numV = 0;
    numI = 0;
    minP.set( FLT_MAX,  FLT_MAX,  FLT_MAX);
    maxP.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);
   
    _skeleton = nullptr;
    _jointMatrices = nullptr;
    _skinMethod = SM_SoftwareSkinning;

    _stateGL = SLGLState::getInstance();  
    _isVolume = true; // is used for RT to decide inside/outside
    _accelStruct = nullptr; // no initial acceleration structure
    _accelStructOutOfDate = true;

    // Add this mesh to the global resource vector for deallocation
    SLScene::current->meshes().push_back(this);
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
    if (P)   delete[] P;    P=nullptr;
    if (N)   delete[] N;    N=nullptr;
    if (C)   delete[] C;    C=nullptr;
    if (T)   delete[] T;    T=nullptr;
    if (Tc)  delete[] Tc;   Tc=nullptr;
    if (Ji)  delete[] Ji;   Ji=nullptr;
    if (Jw)  delete[] Jw;   Jw=nullptr;
    if (I16) delete[] I16;  I16=nullptr;
    if (I32) delete[] I32;  I32=nullptr;

    if (_jointMatrices) delete[] _jointMatrices; _jointMatrices = nullptr;
    if (cpuSkinningP)   delete[] cpuSkinningP; cpuSkinningP = nullptr;
    if (cpuSkinningN)   delete[] cpuSkinningN; cpuSkinningN = nullptr;

    if (_accelStruct) 
    {   delete _accelStruct;      
        _accelStruct = nullptr;
    }

    _bufP.dispose(); 
    _bufN.dispose(); 
    _bufC.dispose(); 
    _bufTc.dispose();
    _bufT.dispose(); 
    _bufI.dispose(); 
    _bufJi.dispose();
    _bufJw.dispose();
    _bufN2.dispose();
    _bufT2.dispose();
}
//-----------------------------------------------------------------------------
//! SLMesh::shapeInit sets the transparency flag of the AABB
void SLMesh::init(SLNode* node)
{   
    if (P && numV && (I16 || I32) && numI)
    {  
        if (!N) calcNormals();

        mat = (mat) ? mat : SLMaterial::defaultMaterial();

        // set transparent flag of the mesh
        if (!node->aabb()->hasAlpha() && mat->hasAlpha()) 
            node->aabb()->hasAlpha(true);
         
        // build tangents for bump mapping
        if (mat->needsTangents() && Tc && T==nullptr)
            calcTangents();
    }
    else
    {   SLstring msg;
        if (!P) msg = "No vertex positions (P)\n";
        if (!numV) msg += "No NO. if vertices (numV)\n";
        if (!I16 && !I32) msg += "No vertex indexes (I16 or I32)\n";
        if (!numI) msg += "No NO. io vertex indexes (numI)\n";
        SL_EXIT_MSG((msg + "in mesh: " + _name).c_str());
    }
}
//-----------------------------------------------------------------------------
/*! 
SLMesh::draw does the OpenGL rendering of the mesh. The GL_TRIANGLES or 
GL_LINES primitives are rendered normally with the vertex position array P,
the normal array N, the array Tc and the index array I16 or I32. 
Optionally you can draw the normals and/or the uniform grid voxels.
<p> The method performs the following steps:</p>
<p>
1) Apply the drawing bits<br>
2) Apply the uniform variables to the shader<br>
2a) Activate a shader program if it is not yet in use and apply all its material paramters.<br>
2b) Pass the modelview and modelview-projection matrix to the shader.<br>
2c) If needed build and pass the inverse modelview and the normal matrix.<br>
2d) If the mesh has a skeleton and HW skinning is applied pass the joint matrices.<br>
3) Build VBOs once<br>
4) Bind and enable attribute pointers<br>
5) Finally do the draw call<br>
6) Disable attribute pointers<br>
7) Draw optional normals & tangents<br>
8) Draw optional acceleration structure<br>
</p>
Please view also the full process of rendering <a href="md_on_paint.html"><b>one frame</b></a>
*/
void SLMesh::draw(SLSceneView* sv, SLNode* node)
{  
    if (P && numV && (I16 || I32) && numI)
    {     
        ////////////////////////
        // 1) Apply Drawing Bits
        ////////////////////////
            
        // Return if hidden
        if (sv->drawBit(SL_DB_HIDDEN) || node->drawBit(SL_DB_HIDDEN)) 
            return;

        SLPrimitive primitiveType = _primitive;
              
        // Set polygon mode
        if (sv->drawBit(SL_DB_WIREMESH) || node->drawBit(SL_DB_WIREMESH))
        {
            #if defined(SL_GLES2)
            primitiveType = SL_LINE_LOOP; // There is no polygon mode on ES2!
            #else
            _stateGL->polygonLine(true);
            #endif
        } else _stateGL->polygonLine(false);

        // Set face culling
        bool noFaceCulling = sv->drawBit(SL_DB_CULLOFF) || node->drawBit(SL_DB_CULLOFF);
        _stateGL->cullFace(!noFaceCulling);
      
        // check if texture exists
        SLbool useTexture = Tc && !sv->drawBit(SL_DB_TEXOFF) && !node->drawBit(SL_DB_TEXOFF);
                                     
        // enable polygon offset if voxels are drawn to avoid stitching
        if (sv->drawBit(SL_DB_VOXELS) || node->drawBit(SL_DB_VOXELS))
            _stateGL->polygonOffset(true, 1.0f, 1.0f);


       
        /////////////////////////////
        // 2) Apply Uniform Variables
        /////////////////////////////

        // 2.a) Apply mesh material if exists & differs from current
        if (mat != SLMaterial::current || SLMaterial::current->program()==nullptr)
            mat->activate(_stateGL, *node->drawBits());
            
        // 2.b) Pass the matrices to the shader program
        SLGLProgram* sp = SLMaterial::current->program();
        sp->uniformMatrix4fv("u_mvMatrix",    1, (SLfloat*)&_stateGL->modelViewMatrix);
        sp->uniformMatrix4fv("u_mvpMatrix",   1, (SLfloat*)_stateGL->mvpMatrix());

        // 2.c) Build & pass inverse, normal & texture matrix only if needed
        SLint locIM = sp->getUniformLocation("u_invMvMatrix");
        SLint locNM = sp->getUniformLocation("u_nMatrix");
        SLint locTM = sp->getUniformLocation("u_tMatrix");

        if (locIM>=0 && locNM>=0) 
        {   _stateGL->buildInverseAndNormalMatrix();
            sp->uniformMatrix4fv(locIM, 1, (SLfloat*)_stateGL->invModelViewMatrix());
            sp->uniformMatrix3fv(locNM, 1, (SLfloat*)_stateGL->normalMatrix());
        } else 
        if (locIM>=0) 
        {   _stateGL->buildInverseMatrix();
            sp->uniformMatrix4fv(locIM, 1, (SLfloat*)_stateGL->invModelViewMatrix());
        } else
        if (locNM>=0) 
        {   _stateGL->buildNormalMatrix();
            sp->uniformMatrix3fv(locNM, 1, (SLfloat*)_stateGL->normalMatrix());
        }
        if (locTM>=0)
        {   if (mat->has3DTexture() && mat->textures()[0]->autoCalcTM3D())
                 calcTex3DMatrix(node);
            else _stateGL->textureMatrix = mat->textures()[0]->tm();
            sp->uniformMatrix4fv(locTM, 1, (SLfloat*)&_stateGL->textureMatrix);
        }

        // 2.d) Do GPU skinning for animated meshes
        if (_skeleton && Ji && Jw && _skinMethod == SM_HardwareSkinning)
        {
            if (!_jointMatrices)
            _jointMatrices = new SLMat4f[_skeleton->numJoints()];

            if (_skeleton->changed())
            {
                // update the joint matrix array
                _skeleton->getJointWorldMatrices(_jointMatrices);
                // remove the changed flag from the skeleton since our joint matrices are up to date again
                // @note    the skeleton referenced here would be the skeleton instance proposed in the documentation
                //          of SLSkeleton. So the _jointMatrices array is the only thing that is concerned with the changed flag
                //          however currently we don't have that and multiple meshes need to know if a skeleton changed this frame.
                //          This is why we can't reset the skeleton changed flag at this point in time. If only one entity or mesh required
                //          the information we wouldn't have a problem.
                // notify all nodes that contain this mesh about the change
                notifyParentNodesAABBUpdate();
            }

            // @todo    Secondly: It is a bad idea to keep the joint data in the mesh itself, this prevents us
            //          from instantiating a single mesh with multiple animations. Needs to be addressed ASAP. (see also SLMesh class problems in SLMesh.h at the top)
            //          In short, the solution would be an entity class which is an instance of a mesh (same mesh data) which can be animated. the original buffers
            //          would be in the original mesh and the cpu skinned buffers in the entity. or if gpu skinned it would just be the jointmatrices array that's in the entity.
            SLint locBM = sp->getUniformLocation("u_jointMatrices");
            sp->uniformMatrix4fv(locBM, _skeleton->numJoints(), (SLfloat*)_jointMatrices, false);
        }


        /////////////////////
        // 3) Build VBOs once
        /////////////////////

        if (!_bufP.id())         _bufP.generate(finalP(), numV, 3);
        if (!_bufN.id()  && N)   _bufN.generate(finalN(), numV, 3);
        if (!_bufC.id()  && C)   _bufC.generate(C, numV, 4);
        if (!_bufTc.id() && Tc)  _bufTc.generate(Tc, numV, 2);
        if (!_bufJi.id() && Ji)  _bufJi.generate(Ji, numV, 4);
        if (!_bufJw.id() && Jw)  _bufJw.generate(Jw, numV, 4);
        if (!_bufT.id()  && T)   _bufT.generate(T, numV, 4);
        if (!_bufI.id()  && I16) _bufI.generate(I16, numI, 1, SL_UNSIGNED_SHORT, SL_ELEMENT_ARRAY_BUFFER);
        if (!_bufI.id()  && I32) _bufI.generate(I32, numI, 1, SL_UNSIGNED_INT,   SL_ELEMENT_ARRAY_BUFFER);    


        ////////////////////////////////////////
        // 4) Bind and enable attribute pointers
        ////////////////////////////////////////

        _bufP.bindAndEnableAttrib(sp->getAttribLocation("a_position"));
        if (_bufN.id()) 
            _bufN.bindAndEnableAttrib(sp->getAttribLocation("a_normal"));
        if (_bufC.id()) 
            _bufC.bindAndEnableAttrib(sp->getAttribLocation("a_color"));
        if (_bufTc.id() && useTexture) 
            _bufTc.bindAndEnableAttrib(sp->getAttribLocation("a_texCoord"));
        if (_bufT.id())  
            _bufT.bindAndEnableAttrib(sp->getAttribLocation("a_tangent"));
        if (_bufJi.id())
            _bufJi.bindAndEnableAttrib(sp->getAttribLocation("a_jointIds"));
        if (_bufJw.id())
            _bufJw.bindAndEnableAttrib(sp->getAttribLocation("a_jointWeights"));
   

        ///////////////////////////////
        // 5): Finally do the draw call
        ///////////////////////////////

        _bufI.bindAndDrawElementsAs(primitiveType, numI, 0);

        ////////////////////////////////
        // 6) Disable attribute pointers
        ////////////////////////////////

        _bufP.disableAttribArray();
        if (_bufN.id())  _bufN.disableAttribArray();
        if (_bufC.id())  _bufC.disableAttribArray();
        if (_bufTc.id()) _bufTc.disableAttribArray();
        if (_bufT.id())  _bufT.disableAttribArray();
        if (_bufJi.id()) _bufJi.disableAttribArray();
        if (_bufJw.id()) _bufJw.disableAttribArray();
      
        //////////////////////////////////////
        // 7) Draw optional normals & tangents
        //////////////////////////////////////

        // All helper lines must be drawn without blending
        SLbool blended = _stateGL->blend();
        if (blended) _stateGL->blend(false);

        if (N && (sv->drawBit(SL_DB_NORMALS) || node->drawBit(SL_DB_NORMALS)))
        {  
            // scale factor r 5% from scaled radius for normals & tangents
            // build array between vertex and normal target point
            float r = node->aabb()->radiusOS() * 0.05f;
            SLVec3f* V2 = new SLVec3f[numV*2];
            for (SLuint i=0; i < numV; ++i)
            {   V2[i<<1] = finalP()[i];
                V2[(i<<1)+1].set(finalP()[i] + finalN()[i]*r);
            }

            // Create or update buffer object for normals
            if (_bufN2.id())
                 _bufN2.update(V2, numV*2, 0);
            else _bufN2.generate(V2, numV*2, 3);

            if (T)
            {   for (SLuint i=0; i < numV; ++i)
                {  V2[(i<<1)+1].set(finalP()[i].x+T[i].x*r,
                                    finalP()[i].y+T[i].y*r,
                                    finalP()[i].z+T[i].z*r);
                }

                // Create or update buffer object for tangents
                if (_bufT2.id())
                     _bufT2.update(V2, numV*2, 3);
                else _bufT2.generate(V2, numV*2, 3);
            }
            delete[] V2;

            _bufN2.drawArrayAsConstantColorLines(SLCol3f::BLUE);
            if (T) _bufT2.drawArrayAsConstantColorLines(SLCol3f::RED);
            if (blended) _stateGL->blend(false);
        } 
        else
        {   // release buffer objects for normal & tangent rendering
            if (_bufN2.id()) _bufN2.dispose();
            if (_bufT2.id()) _bufT2.dispose();
        }

        //////////////////////////////////////////
        // 8) Draw optional acceleration structure
        //////////////////////////////////////////

        if (_accelStruct) 
        {   if (sv->drawBit(SL_DB_VOXELS) || node->drawBit(SL_DB_VOXELS))
            {   _accelStruct->draw(sv);
                _stateGL->polygonOffset(false);
            } else
            {  // Delete the visualization VBO if not rendered anymore
                _accelStruct->disposeBuffers();
            }
        }

        ////////////////////////
        // 6: Draw selected mesh
        ////////////////////////
      
        if (SLScene::current->selectedMesh()==this)
        {   _stateGL->polygonOffset(true, 1.0f, 1.0f);
            _bufP.drawArrayAsConstantColorPoints(SLCol4f::YELLOW, 2);
            _stateGL->polygonLine(false);
            _stateGL->polygonOffset(false);
        } 

        if (blended) _stateGL->blend(true);
            
        GET_GL_ERROR;
    }
    else
    {   SLstring msg;
        if (!P) msg = "No vertex positions (P)\n";
        if (!numV) msg += "No NO. if vertices (numV)\n";
        if (!I16 && !I32) msg += "No vertex indexes (I16 or I32)\n";
        if (!numI) msg += "No NO. io vertex indexes (numI)\n";
        SL_EXIT_MSG((msg + "in mesh: " + _name).c_str());
    }
}
//-----------------------------------------------------------------------------
/*!
SLMesh::hit does the ray-mesh intersection test. If no acceleration 
structure is defined all triangles are tested in a brute force manner.
*/
SLbool SLMesh::hit(SLRay* ray, SLNode* node)
{  
    if (_primitive != SL_TRIANGLES)
        return false;
    
    // Avoid intersection of another mesh before the ray left this one
    if (!ray->isOutside && ray->originNode != node) 
        return false;
     
    if (_accelStruct)
        return _accelStruct->intersect(ray, node);
    else
    {   // intersect against all faces
        SLbool wasHit = false; 

        for (SLuint t=0; t<numI; t+=3)
            if(hitTriangleOS(ray, node, t) && !wasHit) 
                wasHit = true;

        return wasHit;
    }
}
//-----------------------------------------------------------------------------
/*! 
SLMesh::updateStats updates the parent node statistics.
*/
void SLMesh::addStats(SLNodeStats &stats)
{  
    stats.numBytes += sizeof(SLMesh);               // myself
    if (P)  stats.numBytes += numV*sizeof(SLVec3f); // P
    if (N)  stats.numBytes += numV*sizeof(SLVec3f); // N
    if (C)  stats.numBytes += numV*sizeof(SLCol4f); // C
    if (T)  stats.numBytes += numV*sizeof(SLVec3f); // T
    if (Tc) stats.numBytes += numV*sizeof(SLVec2f); // Tc

    if (I16)
         stats.numBytes += numI*sizeof(SLushort);   // I16
    else stats.numBytes += numI*sizeof(SLuint);     // I32
        
    stats.numMeshes++;
    if (_primitive==SL_TRIANGLES) stats.numTriangles += numI/3;
    if (_primitive==SL_LINES)     stats.numLines     += numI/2;
    if (typeid(*this)==typeid(SLLightSphere)) stats.numLights++;
    if (typeid(*this)==typeid(SLLightRect  )) stats.numLights++;

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
    minP.set( FLT_MAX,  FLT_MAX,  FLT_MAX);
    maxP.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // calc min and max point of all vertices
    for (SLuint i=0; i<numV; ++i)
    {   if (finalP()[i].x < minP.x) minP.x = finalP()[i].x;
        if (finalP()[i].x > maxP.x) maxP.x = finalP()[i].x;
        if (finalP()[i].y < minP.y) minP.y = finalP()[i].y;
        if (finalP()[i].y > maxP.y) maxP.y = finalP()[i].y;
        if (finalP()[i].z < minP.z) minP.z = finalP()[i].z;
        if (finalP()[i].z > maxP.z) maxP.z = finalP()[i].z;
    } 
}
//-----------------------------------------------------------------------------
/*!
SLMesh::calcCenterRad calculates the center and the radius of an almost minimal
bounding sphere. Code by Jack Ritter from Graphic Gems.
*/
void SLMesh::calcCenterRad(SLVec3f& center, SLfloat& radius)
{
    SLuint    i;
    SLfloat  dx, dy, dz;
    SLfloat  radius2, xspan, yspan, zspan, maxspan;
    SLfloat  old_to_p, old_to_p_sq, old_to_new;
    SLVec3f  xmin, xmax, ymin, ymax, zmin, zmax, dia1, dia2;

    // FIRST PASS: find 6 minima/maxima points
    xmin.x = ymin.y = zmin.z=  FLT_MAX;
    xmax.x = ymax.y = zmax.z= -FLT_MAX;
   
    for (i=0; i<numV; ++i)
    {
        if (P[i].x < xmin.x) xmin = P[i]; else
        if (P[i].x > xmax.x) xmax = P[i];
        if (P[i].y < ymin.y) ymin = P[i]; else
        if (P[i].y > ymax.y) ymax = P[i];
        if (P[i].z < zmin.z) zmin = P[i]; else
        if (P[i].z > zmax.z) zmax = P[i];
    }
   
    // Set xspan = distance between the 2 points xmin & xmax (squared)
    dx = xmax.x - xmin.x;
    dy = xmax.y - xmin.y;
    dz = xmax.z - xmin.z;
    xspan = dx*dx + dy*dy + dz*dz;

    // Same for y & z spans
    dx = ymax.x - ymin.x;
    dy = ymax.y - ymin.y;
    dz = ymax.z - ymin.z;
    yspan = dx*dx + dy*dy + dz*dz;

    dx = zmax.x - zmin.x;
    dy = zmax.y - zmin.y;
    dz = zmax.z - zmin.z;
    zspan = dx*dx + dy*dy + dz*dz;

    // Set points dia1 & dia2 to the maximally separated pair
    dia1 = xmin; dia2 = xmax; // assume xspan biggest
    maxspan = xspan;
    if (yspan > maxspan)
    {   maxspan = yspan;
        dia1 = ymin; dia2 = ymax;
    }
    if (zspan > maxspan)
    {  dia1 = zmin; dia2 = zmax;
    }

    // dia1,dia2 is a diameter of initial sphere
    // calc initial center
    center.x = (dia1.x + dia2.x)*0.5f;
    center.y = (dia1.y + dia2.y)*0.5f;
    center.z = (dia1.z + dia2.z)*0.5f;
   
    // calculate initial radius*radius and radius
    dx = dia2.x - center.x; // x component of radius vector
    dy = dia2.y - center.y; // y component of radius vector
    dz = dia2.z - center.z; // z component of radius vector
    radius2 = dx*dx + dy*dy + dz*dz;
    radius  = sqrt(radius2);

    // SECOND PASS: increment current sphere
    for (i=0; i<numV; ++i)
    {
        dx = P[i].x - center.x;
        dy = P[i].y - center.y;
        dz = P[i].z - center.z;
        old_to_p_sq = dx*dx + dy*dy + dz*dz;
      
        if (old_to_p_sq > radius2) 	// do r**2 test first
        {
            // this point is outside of current sphere
            old_to_p = sqrt(old_to_p_sq);
         
            // calc radius of new sphere
            radius  = (radius + old_to_p) * 0.5f;
            radius2 = radius*radius; 	// for next r**2 compare
            old_to_new = old_to_p - radius;
         
            // calc center of new sphere
            center.x = (radius*center.x + old_to_new*P[i].x) / old_to_p;
            center.y = (radius*center.y + old_to_new*P[i].y) / old_to_p;
            center.z = (radius*center.z + old_to_new*P[i].z) / old_to_p;
         
            // Suppress if desired
            SL_LOG("\n New sphere: center,radius = %f %f %f   %f",
                   center.x,center.y,center.z, radius);
        }
    }
}
//-----------------------------------------------------------------------------
/*! 
SLMesh::buildAABB builds the passed axis-aligned bounding box in OS and updates
the min & max points in WS with the passed WM of the node.
*/
void SLMesh::buildAABB(SLAABBox &aabb, SLMat4f wmNode)
{   
    // update accel struct and calculate min max
    //updateAccelStruct();
    if (_skeleton)
    {
        minP = _skeleton->minOS();
        maxP = _skeleton->maxOS();
    }     
    else
        calcMinMax();
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
    SLVec3f distMinMax = maxP - minP;
    SLfloat addon = distMinMax.length() * 0.005f;
    minP -= addon;
    maxP += addon;

    if (_accelStruct == nullptr && _primitive == SL_TRIANGLES)
        _accelStruct = new SLUniformGrid(this);

    if (_accelStruct && numI > 15)
    {   _accelStruct->build(minP, maxP);
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
    assert(numV && P && (I16 || I32));

    // Create array for the normals & Zero out the normals array
    delete[] N;
    N = 0;
    _bufN.dispose();

    if (_primitive != SL_TRIANGLES)
        return;

    N = new SLVec3f[numV];
    for (SLuint i = 0; i < numV; ++i) N[i] = SLVec3f::ZERO;
   
    if (I16)
    {   
        // Loop over all triangles
        for (SLuint i = 0; i < numI-3; i+=3)
        {  
            // Calculate the face's normal
            SLVec3f e1, e2, n;
      
            // Calculate edges of triangle
            e1.sub(P[I16[i+1]],P[I16[i+2]]);    // e1 = B - C
            e2.sub(P[I16[i+1]],P[I16[i  ]]);    // e2 = B - A
      
            // Build normal with cross product but do NOT normalize it.
            n.cross(e1,e2);                     // n = e1 x e2

            // Add this normal to its vertices normals
            N[I16[i  ]] += n;
            N[I16[i+1]] += n;
            N[I16[i+2]] += n;
        }
    }
   
    if (I32)
    {   
        for (SLuint i = 0; i < numI-3; i+=3)
        {  
            // Calculate the face's normal
            SLVec3f e1, e2, n;
      
            // Calculate edges of triangle
            e1.sub(P[I32[i+1]],P[I32[i+2]]);    // e1 = B - C
            e2.sub(P[I32[i+1]],P[I32[i  ]]);    // e2 = B - A
      
            // Build normal with cross product but do NOT normalize it.
            n.cross(e1,e2);                     // n = e1 x e2

            // Add this normal to its vertices normals
            N[I32[i  ]] += n;
            N[I32[i+1]] += n;
            N[I32[i+2]] += n;
        }
    } 
      
    // normalize vertex normals
    for (SLuint i=0; i < numV; ++i) N[i].normalize();
}
//-----------------------------------------------------------------------------
//! SLMesh::calcTangents computes the tangents per vertex for triangle meshes. 
/*! SLMesh::calcTangents computes the tangent and bi-tangent per vertex used 
for GLSL normal map bump mapping. The code and mathematical derivation is in 
detail explained in: http://www.terathon.com/code/tangent.html
*/
void SLMesh::calcTangents()
{
    if (P && N && Tc && (I16||I32))
    {
        // Delete old tangents
        delete[] T;
        T = 0;
        
        if (_primitive != SL_TRIANGLES)
            return;
        
        // allocate tangents
        T = new SLVec4f[numV];
      
        // allocate temp arrays for tangents
        SLVec3f* T1 = new SLVec3f[numV * 2];
        SLVec3f* T2 = T1 + numV;
        memset(T1, 0, numV * sizeof(SLVec3f) * 2);

        SLuint iVA, iVB, iVC;
        SLuint numT = numI / 3; //NO. of triangles

        for (SLuint t = 0; t < numT; ++t)
        {
            SLuint i = t * 3;   // vertex index

            // Get the 3 vertex indexes
            if (I16)
            {   iVA = I16[i  ];
                iVB = I16[i+1];
                iVC = I16[i+2];
            } else
            {   iVA = I32[i  ];
                iVB = I32[i+1];
                iVC = I32[i+2];
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

            float r = 1.0F / (s1*t2 - s2*t1);
            SLVec3f sdir((t2*x1 - t1*x2) * r, (t2*y1 - t1*y2) * r, (t2*z1 - t1*z2) * r);
            SLVec3f tdir((s1*x2 - s2*x1) * r, (s1*y2 - s2*y1) * r, (s1*z2 - s2*z1) * r);

            T1[iVA] += sdir;
            T1[iVB] += sdir;
            T1[iVC] += sdir;

            T2[iVA] += tdir;
            T2[iVB] += tdir;
            T2[iVC] += tdir;
        }


        for (SLuint i=0; i < numV; ++i)
        {
            // Gram-Schmidt orthogonalization
            T[i] = T1[i] - N[i] * N[i].dot(T1[i]);
            T[i].normalize();
           
            // Calculate temp. bitangent and store its handedness in T.w
            SLVec3f bitangent;
            bitangent.cross(N[i], T1[i]);
            T[i].w = (bitangent.dot(T2[i]) < 0.0f) ? -1.0f : 1.0f;
        }
       
        delete[] T1;
    }
}
//-----------------------------------------------------------------------------
/* Calculate the texture matrix for 3D texture mapping from the AABB so that
the texture volume surrounds the AABB centrically.
*/
void SLMesh::calcTex3DMatrix(SLNode* node)
{
    SLVec3f min = node->aabb()->minOS();
    SLVec3f max = node->aabb()->maxOS();
    SLVec3f ctr = node->aabb()->centerOS();
    SLVec3f ext = max - ctr;

    // determine the scale factor s from the max. AABB extension 
    SLint dim;
    SLfloat maxExt = ext.maxXYZ(dim);
    SLfloat s = 1.0f/(2.0f * maxExt);

    // scale and translate the texture matrix
    SLGLState* stateGL = SLGLState::getInstance();
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
    #if _DEBUG
    ++SLRay::tests;
    #endif
 
    if (_primitive != SL_TRIANGLES)
        return false;

    // prevent self-intersection of triangle
    if(ray->originMesh == this && ray->originTria == iT) 
        return false;
      
    SLVec3f A, B, C;     // corners
    SLVec3f e1, e2;      // edge 1 and 2
    SLVec3f AO, K, Q;
   
    // get the corner vertices
    if (I16)
    {   A = finalP()[I16[iT  ]];
        B = finalP()[I16[iT+1]];
        C = finalP()[I16[iT+2]];
    } else
    {   A = finalP()[I32[iT  ]];
        B = finalP()[I32[iT+1]];
        C = finalP()[I32[iT+2]];
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
    {   // check only front side triangles           
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
        if (v < 0.0f || u+v > det) return false;
      
        // calculate intersection distance t
        inv_det = 1.0f / det;
        t = e2.dot(Q) * inv_det;
   
        // if intersection is closer replace ray intersection parameters
        if (t > ray->length || t < 0.0f) return false;

        ray->length = t;
      
        // scale down u & v so that u+v<=1
        ray->hitU = u * inv_det;
        ray->hitV = v * inv_det;
    }
    else 
    {   // check front & backside triangles
        if (det < FLT_EPSILON && det > -FLT_EPSILON) return false;
      
        inv_det = 1.0f / det;
      
        // calculate distance from A to ray origin
        AO.sub(ray->originOS, A);
      
        // Calculate barycentric coordinates: u>0 && v>0 && u+v<=1
        u = AO.dot(K) * inv_det;
        if (u < 0.0f || u > 1.0) return false;
      
        // prepare to test v parameter
        Q.cross(AO, e1);
      
        // calculate v parameter and test bounds
        v = Q.dot(ray->dirOS) * inv_det;
        if (v < 0.0f || u+v > 1.0f) return false;
      
        // calculate t, ray intersects triangle
        t = e2.dot(Q) * inv_det;
         
        // if intersection is closer replace ray intersection parameters
        if (t > ray->length || t < 0.0f) return false;

        ray->length = t;
        ray->hitU = u;
        ray->hitV = v;
    }

    ray->hitTriangle = iT;
    ray->hitNode = node;
    ray->hitMesh = this;
    ray->hitMat = (mat) ? mat : SLMaterial::defaultMaterial();

    #if _DEBUG
    ++SLRay::intersections;
    #endif

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
    assert (_primitive == SL_TRIANGLES);

    // Get the triangle indexes
    SLuint iA, iB, iC;
    if (I16)
    {   iA = I16[ray->hitTriangle  ];
        iB = I16[ray->hitTriangle+1];
        iC = I16[ray->hitTriangle+2];
    } else
    {   iA = I32[ray->hitTriangle  ];
        iB = I32[ray->hitTriangle+1];
        iC = I32[ray->hitTriangle+2];
    }
   
    // calculate the hit point in world space
    ray->hitPoint.set(ray->origin + ray->length * ray->dir);
      
    // calculate the interpolated normal with vertex normals in object space
    ray->hitNormal.set(finalN()[iA] * (1-(ray->hitU+ray->hitV)) +
                       finalN()[iB] * ray->hitU +
                       finalN()[iC] * ray->hitV);
                      
    // transform normal back to world space
    ray->hitNormal.set(ray->hitNode->updateAndGetWMN() * ray->hitNormal);
                 
    // invert normal if the ray is inside a mesh
    if (!ray->isOutside) ray->hitNormal *= -1;
   
    // for shading the normal is expected to be unit length
    ray->hitNormal.normalize();
   
    // calculate interpolated texture coordinates
    SLVGLTexture& textures = ray->hitMat->textures();
    if (textures.size() > 0)
    {   SLVec2f Tu(Tc[iB] - Tc[iA]);
        SLVec2f Tv(Tc[iC] - Tc[iA]);
        SLVec2f tc(Tc[iA] + ray->hitU*Tu + ray->hitV*Tv);
        ray->hitTexCol.set(textures[0]->getTexelf(tc.x,tc.y));
      
        // bump mapping
        if (textures.size() > 1)
        {   if (T)
            {  
                // calculate the interpolated tangent with vertex tangent in object space
                SLVec4f hitT(T[iA] * (1-(ray->hitU+ray->hitV)) + 
                             T[iB] * ray->hitU + 
                             T[iC] * ray->hitV);
                         
                SLVec3f T3(hitT.x,hitT.y,hitT.z);         // tangent with 3 components
                T3.set(ray->hitNode->updateAndGetWMN()*T3); // transform tangent back to world space
                SLVec2f d = textures[1]->dsdt(tc.x,tc.y); // slope of bumpmap at tc
                SLVec3f N = ray->hitNormal;               // unperturbated normal
                SLVec3f B(N^T3);                          // binormal tangent B
                B *= T[iA].w;                             // correct handedness
                SLVec3f D(d.x*T3 + d.y*B);                // perturbation vector D
                N+=D;
                N.normalize();
                ray->hitNormal.set(N);
            }
        }
    }
}

//-----------------------------------------------------------------------------
/*! Adds an joint weight to the specified vertex id (max 4 weights per vertex)
returns true if added successfully, false if already full
*/
SLbool SLMesh::addWeight(SLint vertId, SLuint jointId, SLfloat weight)
{
    if (!Ji || !Jw)
        return false;

    assert(vertId < (SLint)numV && "An illegal index was passed in to SLMesh::addWeight");
    
    SLVec4f& jId = Ji[vertId];
    SLVec4f& jWeight = Jw[vertId];

    // "iterator" over our vectors
    SLfloat* jIdIt = &jId.x;
    SLfloat* jWeightIt = &jWeight.x;

    for (SLint i = 0; i < 4; ++i)
    {
        if (*jWeightIt == 0.0f)
        {
            *jIdIt = (SLfloat)jointId;
            *jWeightIt = weight;
            break;
        }

        jIdIt++;
        jWeightIt++;
    }

    return true;
}

//-----------------------------------------------------------------------------
/*! Sets the current skinning method.
@todo   This function is still kind of hackish, we manually change the material in the mesh. The skinning
        shouldn't rely on a material however, the shader should know if it needs skinning variables and 
        transformations and apply them to the current used shader.
*/
void SLMesh::skinMethod(SLSkinMethod method) 
{ 
    if (method == _skinMethod)
        return;

    // return if this isn't a skinned mesh
    if (!_skeleton || !Ji || !Jw)
        return;

    _skinMethod = method;

    if (_skinMethod == SM_HardwareSkinning)
    {
        _finalP = &P;
        _finalN = &N;

        // if we are a textured mesh
        if (Tc)
        {   SLGLGenericProgram* skinningShaderTex = new SLGLGenericProgram("PerPixBlinnTexSkinned.vert",
                                                                           "PerPixBlinnTex.frag");
            mat->program(skinningShaderTex);
        } else
        {   SLGLGenericProgram* skinningShader = new SLGLGenericProgram("PerVrtBlinnSkinned.vert",
                                                                        "PerVrtBlinn.frag");
            mat->program(skinningShader);
        }
    }
    else
    {
        mat->program(0);
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
    if (!cpuSkinningP)
    {   cpuSkinningP = new SLVec3f[numV];
        for (SLuint i = 0; i < numV; ++i)
            cpuSkinningP[i] = P[i];
    }
    if (!cpuSkinningN && N) 
    {   cpuSkinningN = new SLVec3f[numV];
        for (SLuint i = 0; i < numV; ++i)
            cpuSkinningN[i] = N[i];
    }

    // Create array for joint matrices once
    if (!_jointMatrices)
        _jointMatrices = new SLMat4f[_skeleton->numJoints()];

    // update the joint matrix array
    _skeleton->getJointWorldMatrices(_jointMatrices);

    notifyParentNodesAABBUpdate();

    // temporarily set finalP and finalN
    _finalP = &cpuSkinningP;
    _finalN = &cpuSkinningN;
    
    // flag acceleration structure to be rebuilt
    _accelStructOutOfDate = true;
        
    // iterate over all vertices and write to new buffers
    for (SLuint i = 0; i < numV; ++i)
    {
        cpuSkinningP[i] = SLVec3f::ZERO;
        if (N) cpuSkinningN[i] = SLVec3f::ZERO;

        // array form for easier iteration
        SLfloat jointWeights[4] = {Jw[i].x, Jw[i].y, Jw[i].z, Jw[i].w};
        SLint   jointIndices[4] = {(SLint)Ji[i].x,
                                   (SLint)Ji[i].y,
                                   (SLint)Ji[i].z,
                                   (SLint)Ji[i].w};
                    
        // accumulate final normal and positions
        for (SLint j = 0; j < 4; ++j)
        {   if (jointWeights[j] > 0.0f)
            {
                const SLMat4f& jm = _jointMatrices[jointIndices[j]];
                SLVec4f tempPos = jm * P[i];
                cpuSkinningP[i].x += tempPos.x * jointWeights[j];
                cpuSkinningP[i].y += tempPos.y * jointWeights[j];
                cpuSkinningP[i].z += tempPos.z * jointWeights[j];

                if (N) 
                {   // Build the 3x3 submatrix in GLSL 110 (= mat3 jt3 = mat3(jt))
                    // for the normal transform that is the normally the inverse transpose.
                    // The inverse transpose can be ignored as long as we only have
                    // rotation and uniform scaling in the 3x3 submatrix.
                    SLMat3f jnm = jm.mat3();
                    //jnm.invert();
                    //jnm.transpose();
                    cpuSkinningN[i] += jnm * N[i] * jointWeights[j];
                }
            }
        }
    }  

    // update or create buffers
    if (_bufP.id()) 
         _bufP.update(finalP(), numV, 0);
    else _bufP.generate(finalP(), numV, 3);
    if (N)
    {   if (_bufN.id())
             _bufN.update(finalN(), numV, 0);
        else _bufN.generate(finalN(), numV, 3);
    }
}

//-----------------------------------------------------------------------------
void SLMesh::notifyParentNodesAABBUpdate() const
{
    SLVNode nodes = SLScene::current->root3D()->findChildren(this);
    for (auto node : nodes)
        node->needAABBUpdate();
}
//-----------------------------------------------------------------------------
