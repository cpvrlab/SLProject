//#############################################################################
//  File:      SL/SLAssimpImporter.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

// @todo    add log output to the other parts of the importer

#include <stdafx.h>
#include <SLAssimpImporter.h>
#include <SLScene.h>
#include <SLGLTexture.h>
#include <SLMaterial.h>
#include <SLSkeleton.h>
#include <SLAnimation.h>
#include <SLGLProgram.h>

// assimp is only included in the source file to not expose it to the rest of the framework
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


struct KeyframeData
{
    KeyframeData()
    : translation(NULL), 
    rotation(NULL),
    scaling(NULL)
    { } 

    KeyframeData(aiVectorKey* trans, aiQuatKey* rot, aiVectorKey* scl)
    {
        translation = trans;
        rotation = rot;
        scaling = scl;
    }
    
    aiVectorKey* translation;
    aiQuatKey* rotation;
    aiVectorKey* scaling;
};
typedef std::map<SLfloat, KeyframeData> KeyframeMap;



// helper functions
// @todo we can probably make the getTranslation, getRotation and getScaling functions smaller by putting the common
//       functionality in a sepearte function

/*  Get the correct translation out of the keyframes map for a given time
    this function interpolates linearly if no value is present in the 

    @note    this function does not wrap around to interpolate. if there is no 
             translation key to the right of the passed in time then this function will take
             the last known value on the left!
*/
SLVec3f getTranslation(SLfloat time, const KeyframeMap& keyframes)
{
    KeyframeMap::const_iterator it = keyframes.find(time);
    aiVector3D result(0, 0, 0); // return 0 position of nothing was found

    // If the timesamp passed in doesnt exist then something in the loading of the kfs went wrong
    assert(it != keyframes.end() && "A KeyframeMap was passed in with an illegal timestamp."); // @todo this should throw an exception and not kill the app

    aiVectorKey* transKey = it->second.translation;

    // the timestamp has a valid translation value, just return the SL type
    if (transKey)
        result = transKey->mValue;
    else
    {
        aiVectorKey* frontKey = NULL;
        aiVectorKey* backKey = NULL;

        // no translation value present, we must interpolate
        KeyframeMap::const_reverse_iterator revIt(it);

        // search to the right
        for (; it != keyframes.end(); it++)
        {
            if (it->second.translation != NULL)
            {
                backKey = it->second.translation;
                break;
            }
        }

        // search to the left
        for (; revIt != keyframes.rend(); revIt++)
        {
            if (revIt->second.translation != NULL)
            {
                frontKey = revIt->second.translation;
                break;
            }
        }

        if (frontKey && backKey) 
        {
            SLfloat frontTime = revIt->first;
            SLfloat backTime = it->first;
            SLfloat t = (time - frontTime) / (backTime - frontTime);

            result = frontKey->mValue + (t * (backKey->mValue - frontKey->mValue));
        }
        else if (frontKey) 
        {
            result = frontKey->mValue;
        }
        else if (backKey)
        {
            result = backKey->mValue;
        }
    }
    
    return SLVec3f(result.x, result.y, result.z);
}

/*  Get the correct translation out of the keyframes map for a given time
    this function interpolates linearly if no value is present in the 

    @note    this function does not wrap around to interpolate. if there is no 
             translation key to the right of the passed in time then this function will take
             the last known value on the left!
*/
SLVec3f getScaling(SLfloat time, const KeyframeMap& keyframes)
{
    KeyframeMap::const_iterator it = keyframes.find(time);
    aiVector3D result(1, 1, 1); // return unit scale if no kf was found

    // If the timesamp passed in doesnt exist then something in the loading of the kfs went wrong
    assert(it != keyframes.end() && "A KeyframeMap was passed in with an illegal timestamp."); // @todo this should throw an exception and not kill the app

    aiVectorKey* scaleKey = it->second.scaling;

    // the timestamp has a valid translation value, just return the SL type
    if (scaleKey)
        result = scaleKey->mValue;
    else
    {
        aiVectorKey* frontKey = NULL;
        aiVectorKey* backKey = NULL;

        // no translation value present, we must interpolate
        KeyframeMap::const_reverse_iterator revIt(it);

        // search to the right
        for (; it != keyframes.end(); it++)
        {
            if (it->second.rotation != NULL)
            {
                backKey = it->second.scaling;
                break;
            }
        }

        // search to the left
        for (; revIt != keyframes.rend(); revIt++)
        {
            if (revIt->second.rotation != NULL)
            {
                frontKey = revIt->second.scaling;
                break;
            }
        }

        if (frontKey && backKey) 
        {
            SLfloat frontTime = revIt->first;
            SLfloat backTime = it->first;
            SLfloat t = (time - frontTime) / (backTime - frontTime);

            result = frontKey->mValue + (t * (backKey->mValue - frontKey->mValue));
        }
        else if (frontKey) 
        {
            result = frontKey->mValue;
        }
        else if (backKey)
        {
            result = backKey->mValue;
        }
    }
    
    return SLVec3f(result.x, result.y, result.z);
}

/*  Get the correct translation out of the keyframes map for a given time
    this function interpolates linearly if no value is present in the 

    @note    this function does not wrap around to interpolate. if there is no 
             translation key to the right of the passed in time then this function will take
             the last known value on the left!
*/
SLQuat4f getRotation(SLfloat time, const KeyframeMap& keyframes)
{
    KeyframeMap::const_iterator it = keyframes.find(time);
    aiQuaternion result(1, 0, 0, 0); // identity rotation

    // If the timesamp passed in doesnt exist then something in the loading of the kfs went wrong
    assert(it != keyframes.end() && "A KeyframeMap was passed in with an illegal timestamp."); // @todo this should throw an exception and not kill the app

    aiQuatKey* rotKey = it->second.rotation;

    // the timestamp has a valid translation value, just return the SL type
    if (rotKey)
        result = rotKey->mValue;
    else
    {
        aiQuatKey* frontKey = NULL;
        aiQuatKey* backKey = NULL;

        // no translation value present, we must interpolate
        KeyframeMap::const_reverse_iterator revIt(it);

        // search to the right
        for (; it != keyframes.end(); it++)
        {
            if (it->second.rotation != NULL)
            {
                backKey = it->second.rotation;
                break;
            }
        }

        // search to the left
        for (; revIt != keyframes.rend(); revIt++)
        {
            if (revIt->second.rotation != NULL)
            {
                frontKey = revIt->second.rotation;
                break;
            }
        }

        if (frontKey && backKey) 
        {
            SLfloat frontTime = revIt->first;
            SLfloat backTime = it->first;
            SLfloat t = (time - frontTime) / (backTime - frontTime);

            aiQuaternion::Interpolate(result, frontKey->mValue, backKey->mValue, t);
        }
        else if (frontKey) 
        {
            result = frontKey->mValue;
        }
        else if (backKey)
        {
            result = backKey->mValue;
        }
    }
    
    return SLQuat4f(result.x, result.y, result.z, result.w);
}

//-----------------------------------------------------------------------------
/*! Loads the scene from a file and creates materials with textures, the 
meshes and the nodes for the scene graph. Materials, textures and meshes are
added to the according vectors of SLScene for later deallocation.
*/
SLNode* SLAssimpImporter::load(SLstring file,        //!< File with path or on default path 
                       SLbool loadMeshesOnly,//!< Only load nodes with meshes
                       SLuint flags)         //!< Import flags (see assimp/postprocess.h)
{
    // Check existance
    if (!SLFileSystem::fileExists(file))
    {   file = defaultPath + file;
        if (!SLFileSystem::fileExists(file))
        {   SLstring msg = "SLAssimpImporter: File not found: " + file + "\n";
            SL_WARN_MSG(msg.c_str());
            return NULL;
        }
    }

    // Import file with assimp importer
    Assimp::Importer ai;
    const aiScene* scene = ai.ReadFile(file.c_str(), (SLuint)flags);
    if (!scene)
    {   SLstring msg = "Failed to load file: " + 
                        file + "\n" + ai.GetErrorString();
        SL_WARN_MSG(msg.c_str());
        return NULL;
    }

    // initial scan of the scene
    performInitialScan(scene);

    // load skeleton
    loadSkeleton(NULL, _skeletonRoot);

    // load materials
    SLstring modelPath = SLUtils::getPath(file);
    SLVMaterial materials;
    for(SLint i = 0; i < (SLint)scene->mNumMaterials; i++)
        materials.push_back(loadMaterial(i, scene->mMaterials[i], modelPath));

    // load meshes & set their material
    std::map<int, SLMesh*> meshMap;  // map from the ai index to our mesh
    for(SLint i = 0; i < (SLint)scene->mNumMeshes; i++)
    {   SLMesh* mesh = loadMesh(scene->mMeshes[i]);
        if (mesh != 0)
        {   mesh->mat = materials[scene->mMeshes[i]->mMaterialIndex];
            _meshes.push_back(mesh);
            meshMap[i] = mesh;
        }
    } 
    
    // add skinned material to the meshes with joint animations
    // @todo: Can we do skinning in the shader without the need to dictated which material to use?
    if (_skinnedMeshes.size() > 0) {
        SLGLGenericProgram* skinningShader = new SLGLGenericProgram("PerVrtBlinnSkinned.vert","PerVrtBlinn.frag");
        SLGLGenericProgram* skinningShaderTex = new SLGLGenericProgram("PerPixBlinnTex.vert","PerPixBlinnTex.frag");
        for (SLint i = 0; i < _skinnedMeshes.size(); i++)
        {
            SLMesh* mesh = _skinnedMeshes[i];
            if (mesh->Tc)
                mesh->mat->program(skinningShaderTex);
            else
                mesh->mat->program(skinningShader);
        }
    }


    // load the scene nodes recursively
    _sceneRoot = loadNodesRec(NULL, scene->mRootNode, meshMap, loadMeshesOnly);

    // load animations
    vector<SLAnimation*> animations;
    for (SLint i = 0; i < (SLint)scene->mNumAnimations; i++)
        animations.push_back(loadAnimation(scene->mAnimations[i]));


    // clear the intermediate data
    clear();

    logMessage(LV_Minimal, "\n---------------------------\n\n");

    return _sceneRoot;
}

//-----------------------------------------------------------------------------
/** Clears all helper containers
*/
void SLAssimpImporter::clear()
{
    _nodeMap.clear();
    _jointOffsets.clear();
    _skeletonRoot = NULL;
    _skeleton = NULL;
    _skinnedMeshes.clear();
}
//-----------------------------------------------------------------------------
/** return an aiNode ptr if name exists, or null if it doesn't */
aiNode* SLAssimpImporter::getNodeByName(const SLstring& name)
{
	if(_nodeMap.find(name) != _nodeMap.end())
		return _nodeMap[name];

	return NULL;
}

//-----------------------------------------------------------------------------
/** return an aiBone ptr if name exists, or null if it doesn't */
const SLMat4f SLAssimpImporter::getOffsetMat(const SLstring& name)
{
	if(_jointOffsets.find(name) != _jointOffsets.end())
		return _jointOffsets[name];

	return SLMat4f();
}

//-----------------------------------------------------------------------------
/** populates nameToNode, nameToBone, jointGroups, skinnedMeshes */
void SLAssimpImporter::performInitialScan(const aiScene* scene)
{
    // populate the _nameToNode map and print the assimp structure on detailed log verbosity.
    logMessage(LV_Detailed, "[Assimp scene]\n");
    logMessage(LV_Detailed, "  Cameras: %d\n", scene->mNumCameras);
    logMessage(LV_Detailed, "  Lights: %d\n", scene->mNumLights);
    logMessage(LV_Detailed, "  Meshes: %d\n", scene->mNumMeshes);
    logMessage(LV_Detailed, "  Materials: %d\n", scene->mNumMaterials);
    logMessage(LV_Detailed, "  Textures: %d\n", scene->mNumTextures);
    logMessage(LV_Detailed, "  Animations: %d\n", scene->mNumAnimations);
    
    logMessage(LV_Detailed, "---------------------------------------------\n");
    logMessage(LV_Detailed, "  Node node tree: \n");
    findNodes(scene->mRootNode, "  ", true);
    
    logMessage(LV_Detailed, "---------------------------------------------\n");
    logMessage(LV_Detailed, "   Searching for skinned meshes and scanning joint names.\n");


    findJoints(scene);
    findSkeletonRoot();
}

//-----------------------------------------------------------------------------
/** scans the assimp scene graph structure and populates nameToNode */
void SLAssimpImporter::findNodes(aiNode* node, SLstring padding, SLbool lastChild)
{ 
    SLstring name = node->mName.C_Str();
    // this should not happen
    assert(_nodeMap.find(name) == _nodeMap.end() && "Duplicated node name found!");
    _nodeMap[name] = node;
    
    //logMessage(LV_Detailed, "%s   |\n", padding.c_str());
    logMessage(LV_Detailed, "%s  |-[%s]   (%d children, %d meshes)\n", 
               padding.c_str(), 
               name.c_str(), 
               node->mNumChildren, 
               node->mNumMeshes);
    
    if (lastChild) padding += "   ";
    else padding += "  |";

    for (SLuint i = 0; i < node->mNumChildren; i++)
    {
        findNodes(node->mChildren[i], padding, (i == node->mNumChildren-1));
    }
}
//-----------------------------------------------------------------------------
/** scans all meshes in the assimp scene and populates nameToBone and jointGroups */
void SLAssimpImporter::findJoints(const aiScene* scene)
{
    for (SLuint i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];
        if(!mesh->HasBones())
            continue;

		logMessage(LV_Normal, "   Mesh '%s' contains %d joints.\n", mesh->mName.C_Str(), mesh->mNumBones);
        
        for (SLuint j = 0; j < mesh->mNumBones; j++)
        {
			SLstring name = mesh->mBones[j]->mName.C_Str();
            std::map<SLstring, SLMat4f>::iterator it = _jointOffsets.find(name);
			if(it != _jointOffsets.end())
				continue;

            // add the offset matrix to our offset matrix map
			SLMat4f offsetMat;
			memcpy(&offsetMat, &mesh->mBones[j]->mOffsetMatrix, sizeof(SLMat4f));
			offsetMat.transpose();
			_jointOffsets[name] = offsetMat;


			logMessage(LV_Detailed, "     Bone '%s' found.\n", name.c_str());
        }
    }
}

//-----------------------------------------------------------------------------
/** finds the common ancestor for each remaining group in jointGroups, these are our final skeleton roots */
void SLAssimpImporter::findSkeletonRoot()
{
    _skeletonRoot = NULL;
    // early out if we don't have any joint bindings
    if (_jointOffsets.size() == 0) return;
    
	vector<NodeList> ancestorList(_jointOffsets.size());
    SLint minDepth = INT_MAX;
    SLint index = 0;    

    logMessage(LV_Detailed, "Building joint ancestor lists.\n");

    JointOffsetMap::iterator it = _jointOffsets.begin();
    for (; it != _jointOffsets.end(); it++, index++)
    {
        aiNode* node = getNodeByName(it->first);
        NodeList& list = ancestorList[index];

        while (node)
        {
            list.insert(list.begin(), node);
            node = node->mParent;
        }

        // log the gathered ancestor list if on diagnostic
        if (LV_Diagnostic)
        {
            logMessage(LV_Diagnostic, "   '%s' ancestor list: ", it->first.c_str());
            for (SLint i = 0; i < list.size(); i++)
                logMessage(LV_Diagnostic, "'%s' ", list[i]->mName.C_Str());
            logMessage(LV_Diagnostic, "\n");
        }
        else
            logMessage(LV_Detailed, "   '%s' lies at a depth of %d\n", it->first.c_str(), list.size());

        minDepth = min(minDepth, (SLint)list.size());
    }


    logMessage(LV_Detailed, "Bone ancestor lists completed, min depth: %d\n", minDepth);
    
    logMessage(LV_Detailed, "Searching ancestor lists for common ancestor.\n");
    // now we have a ancestor list for each joint node beginning with the root node
    for (SLint i = 0; i < minDepth; i++) 
    {
        SLbool failed = false;
        aiNode* lastMatch = ancestorList[0][i];
        for (SLint j = 1; j < ancestorList.size(); j++)
        {
            if (ancestorList[j][i] != lastMatch)
                failed = true;

            lastMatch = ancestorList[j][i];
        }

        // all ancestors matched
        if (!failed)
        {
            _skeletonRoot = lastMatch;
            logMessage(LV_Detailed, "Found matching ancestor '%s'.\n", _skeletonRoot->mName.C_Str());
        }
        else
        {
            break;
        }
    }

    // seems like the above can be wrong, we should just select the common ancestor that is one below the assimps root
    // @todo fix this function up and make sure there exists a second element
    if (!_skeletonRoot)
    _skeletonRoot = ancestorList[0][1];

    logMessage(LV_Normal, "Determined '%s' to be the skeleton's root node.\n", _skeletonRoot->mName.C_Str());
}
//-----------------------------------------------------------------------------
/** Loads the skeleton */
void SLAssimpImporter::loadSkeleton(SLJoint* parent, aiNode* node)
{
    if (!node)
        return;


    SLJoint* joint;
    SLstring name = node->mName.C_Str();
    if (!parent)
    {
	    logMessage(LV_Normal, "Loading skeleton skeleton.\n");
        _skeleton = new SLSkeleton;
        _jointIndex = 0;
        joint = _skeleton->createJoint(name, _jointIndex++); // @todo add a joint creator that also sets the joints name
        _skeleton->root(joint);
    }
    else
    {
        joint = parent->createChild(name, _jointIndex++);
    }

    joint->offsetMat(getOffsetMat(name));
    
    // set the initial state for the joints (in case we render the model without playing its animation)
    // an other possibility is to set the joints to the inverse offset matrix so that the model remains in
    // its bind pose
    // some files will report the node transformation as the animation state transformation that the
    // model had when exporting (in case of our astroboy its in the middle of the animation=
    // it might be more desirable to have ZERO joint transformations in the initial pose
    // to be able to see the model without any joint modifications applied
    // exported state
    
    // set the current node transform as the initial state
    /**
    SLMat4f om;
    memcpy(&om, &node->mTransformation, sizeof(SLMat4f));
    om.transpose();
    joint->om(om);
    joint->setInitialState();
    /*/
    // set the binding pose as initial state
    SLMat4f om;
    om = joint->offsetMat().inverse();
    if (parent)
        om = parent->updateAndGetWM().inverse() * om;
    joint->om(om);
    joint->setInitialState();
    /**/


    for (SLuint i = 0; i < node->mNumChildren; i++)
        loadSkeleton(joint, node->mChildren[i]);
}

//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadMaterial loads the AssImp material an returns the SLMaterial.
The materials and textures are added to the SLScene material and texture 
vectors.
*/
SLMaterial* SLAssimpImporter::loadMaterial(SLint index, 
                         aiMaterial *material, 
                         SLstring modelPath)
{
    // Get the materials name
    aiString matName;
    material->Get(AI_MATKEY_NAME, matName);
    SLstring name = matName.data;
    if (name.empty()) name = "Import Material";
   
    // Create SLMaterial instance. It is also added to the SLScene::_materials vector
    SLMaterial* mat = new SLMaterial(name.c_str());

    // set the texture types to import into our material
    const SLint		textureCount = 4;
    aiTextureType	textureTypes[textureCount];
    textureTypes[0] = aiTextureType_DIFFUSE;
    textureTypes[1] = aiTextureType_NORMALS;
    textureTypes[2] = aiTextureType_SPECULAR;
    textureTypes[3] = aiTextureType_HEIGHT;
   
    // load all the textures for this material and add it to the material vector
    for(SLint i = 0; i < textureCount; ++i) 
    {   if(material->GetTextureCount(textureTypes[i]) > 0) 
        {   aiString aipath;
            material->GetTexture(textureTypes[i], 0, &aipath, NULL, NULL, NULL, NULL, NULL);
            SLTexType texType = textureTypes[i]==aiTextureType_DIFFUSE  ? ColorMap :
                                textureTypes[i]==aiTextureType_NORMALS  ? NormalMap :
                                textureTypes[i]==aiTextureType_SPECULAR ? GlossMap :
                                textureTypes[i]==aiTextureType_HEIGHT   ? HeightMap : 
                                UnknownMap;
            SLstring texFile = checkFilePath(modelPath, aipath.data);
            SLGLTexture* tex = loadTexture(texFile, texType);
            mat->textures().push_back(tex);
        }
    }
   
    // get color data
    aiColor3D ambient, diffuse, specular, emissive;
    SLfloat shininess, refracti, reflectivity, opacity;
    material->Get(AI_MATKEY_COLOR_AMBIENT, ambient);
    material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
    material->Get(AI_MATKEY_COLOR_SPECULAR, specular);
    material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
    material->Get(AI_MATKEY_SHININESS, shininess);
    material->Get(AI_MATKEY_REFRACTI, refracti);
    material->Get(AI_MATKEY_REFLECTIVITY, reflectivity);
    material->Get(AI_MATKEY_OPACITY, opacity);

    // increase shininess if specular color is not low.
    // The material will otherwise be to bright
    if (specular.r > 0.5f &&
        specular.g > 0.5f &&
        specular.b > 0.5f &&
        shininess < 0.01f)
        shininess = 10.0f;

    // set color data
    mat->ambient(SLCol4f(ambient.r, ambient.g, ambient.b));
    mat->diffuse(SLCol4f(diffuse.r, diffuse.g, diffuse.b));
    mat->specular(SLCol4f(specular.r, specular.g, specular.b));
    mat->emission(SLCol4f(emissive.r, emissive.g, emissive.b));
    mat->shininess(shininess);
    //mat->kr(reflectivity);
    //mat->kt(1.0f-opacity);
    //mat->kn(refracti);

    return mat;
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadTexture loads the AssImp texture an returns the SLGLTexture
*/
SLGLTexture* SLAssimpImporter::loadTexture(SLstring& textureFile, SLTexType texType)
{
    SLVGLTexture& sceneTex = SLScene::current->textures();

    // return if a texture with the same file allready exists
    SLbool exists = false;
    for (SLint i=0; i<sceneTex.size(); ++i)
        if (sceneTex[i]->name()==textureFile) 
            return sceneTex[i];

    // Create the new texture. It is also push back to SLScene::_textures
    SLGLTexture* texture = new SLGLTexture(textureFile,
                                           GL_LINEAR_MIPMAP_LINEAR,
                                           GL_LINEAR,
                                           texType);
    return texture;
}

//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadMesh creates a new SLMesh an copies the meshs vertex data and
triangle face indexes. Normals & tangents are not loaded. They are calculated
in SLMesh.
*/
SLMesh* SLAssimpImporter::loadMesh(aiMesh *mesh)
{
    // Count first the NO. of triangles in the mesh
    SLuint numTriangles = 0;
    for(unsigned int i = 0; i <  mesh->mNumFaces; ++i)
        if(mesh->mFaces[i].mNumIndices == 3)
            numTriangles++;

    // We only load meshes that contain triangles
    if (numTriangles==0 || mesh->mNumVertices==0)
        return NULL; 

    // create a new mesh. 
    // The mesh pointer is added automatically to the SLScene::meshes vector.
    SLstring name = mesh->mName.data;
    SLMesh *m = new SLMesh(name.empty() ? "Imported Mesh" : name);

    // create position & normal array
    m->numV = mesh->mNumVertices;
    m->P = new SLVec3f[m->numV];
    m->N = new SLVec3f[m->numV];

    // create texCoord array if needed
    if (mesh->HasTextureCoords(0))
        m->Tc = new SLVec2f[m->numV];

    // copy vertex positions & texCoord
    for(SLuint i = 0; i < m->numV; ++i)
    {   m->P[i].set(mesh->mVertices[i].x, 
        mesh->mVertices[i].y, 
        mesh->mVertices[i].z);
        if (m->N)
            m->N[i].set(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        if (m->Tc)
            m->Tc[i].set(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
    }

    // create face index array
    m->numI = mesh->mNumFaces * 3;
    if (m->numV < 65536)
    {   m->I16 = new SLushort[m->numI];

        // load face triangle indexes only
        SLuint j = 0;
        for(SLuint i = 0; i <  mesh->mNumFaces; ++i)
        {   if(mesh->mFaces[i].mNumIndices == 3)
            {   m->I16[j++] = mesh->mFaces[i].mIndices[0];
                m->I16[j++] = mesh->mFaces[i].mIndices[1];
                m->I16[j++] = mesh->mFaces[i].mIndices[2];
            }
        }
    } else 
    {   m->I32 = new SLuint[m->numI];

        // load face triangle indexes only
        SLuint j = 0;
        for(SLuint i = 0; i <  mesh->mNumFaces; ++i)
        {  if(mesh->mFaces[i].mNumIndices == 3)
            {   m->I32[j++] = mesh->mFaces[i].mIndices[0];
                m->I32[j++] = mesh->mFaces[i].mIndices[1];
                m->I32[j++] = mesh->mFaces[i].mIndices[2];
            }
        }
    }

    //m->calcNormals();

    // load joints
    if (mesh->HasBones())
    {
        _skinnedMeshes.push_back(m);
        m->skeleton(_skeleton);

        m->Ji = new SLVec4f[m->numV];
        m->Jw = new SLVec4f[m->numV];
        
        // make sure to initialize the weights with 0 vectors
        std::fill_n(m->Ji, m->numV, SLVec4f(0, 0, 0, 0));
        std::fill_n(m->Jw, m->numV, SLVec4f(0, 0, 0, 0));

        for (SLuint i = 0; i < mesh->mNumBones; i++)
        {
            aiBone* joint = mesh->mBones[i];
            SLJoint* slJoint = _skeleton->getJoint(joint->mName.C_Str());
            SLuint jointId = slJoint->handle(); // @todo make sure that the returned joint actually exists, else we need to throw here since something in the importer must've gone wrong!

            for (SLuint j = 0; j < joint->mNumWeights; j++)
            {
                // add the weight
                SLuint vertId = joint->mWeights[j].mVertexId;
                SLfloat weight = joint->mWeights[j].mWeight;

                m->addWeight(vertId, jointId, weight);
            }

        }

    }/*
    cout << "imported" << m->name() << "\n";
    cout << "weights: \n";

    for (SLint i = 0; i < m->numV; i++) {
        cout << "   "<< i << "-> (";
            
        if(m->Bw[i].x > 0.0f) cout << m->Bi[i].x << ": " << m->Bw[i].x << ";";
        if(m->Bw[i].y > 0.0f) cout << m->Bi[i].y << ": " << m->Bw[i].y << ";";
        if(m->Bw[i].z > 0.0f) cout << m->Bi[i].z << ": " << m->Bw[i].z << ";";
        if(m->Bw[i].w > 0.0f) cout << m->Bi[i].w << ": " << m->Bw[i].w << ";";
        
        cout << ")\n";
    }*/

    return m;
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadNodesRec loads the scene graph node tree recursively.
*/
SLNode* SLAssimpImporter::loadNodesRec(
   SLNode *curNode,     //!< Pointer to the current node. Pass NULL for root node
   aiNode *node,        //!< The according assimp node. Pass NULL for root node
   SLMeshMap& meshes,   //!< Reference to the meshes vector
   SLbool loadMeshesOnly) //!< Only load nodes with meshes
{
    // we're at the root
    if(!curNode) 
        curNode = new SLNode(node->mName.data);
        
    // load local transform
   aiMatrix4x4* M = &node->mTransformation;
   SLMat4f SLM(M->a1, M->a2, M->a3, M->a4,
                      M->b1, M->b2, M->b3, M->b4,
                      M->c1, M->c2, M->c3, M->c4,
                      M->d1, M->d2, M->d3, M->d4);

   curNode->om(SLM);

    // add the meshes
    for (SLuint i=0; i < node->mNumMeshes; ++i)
    {
        // Only add meshes that were added to the meshMap (triangle meshes)
        if (meshes.count(node->mMeshes[i]))
            curNode->addMesh(meshes[node->mMeshes[i]]);
    }

    // load children recursively
    for(SLuint i = 0; i < node->mNumChildren; i++) 
    {  
        // skip the skeleton
        if (node->mChildren[i] == _skeletonRoot)
            continue;

        // only load children nodes with meshes or children
        if (!loadMeshesOnly || aiNodeHasMesh(node->mChildren[i]))
        {   SLNode *child = new SLNode(node->mChildren[i]->mName.data);
            curNode->addChild(child);
            loadNodesRec(child, node->mChildren[i], meshes);
        }
    }

    return curNode;
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadAnimation loads the scene graph node tree recursively.
*/
// @todo how do we handle multiple skeletons in one file?
SLAnimation* SLAssimpImporter::loadAnimation(aiAnimation* anim)
{
    int animCount = 0;
    if (_skeleton) animCount = _skeleton->numAnimations();
    ostringstream oss;
    oss << "unnamed_anim_" << animCount;
    SLstring animName = oss.str();
    SLfloat animTicksPerSec = (anim->mTicksPerSecond == 0) ? 30.0f : (SLfloat)anim->mTicksPerSecond;
    SLfloat animDuration = (SLfloat)anim->mDuration / animTicksPerSec;

    if (anim->mName.length > 0)
        animName = anim->mName.C_Str();

    // log
    logMessage(LV_Minimal, "\nLoading animation %s\n", animName.c_str());
    logMessage(LV_Normal, " Duration(seconds): %f \n", animDuration);
    logMessage(LV_Normal, " Duration(ticks): %f \n", anim->mDuration);
    logMessage(LV_Normal, " Ticks per second: %f \n", animTicksPerSec);
    logMessage(LV_Normal, " Num channels: %d\n", anim->mNumChannels);
                
    // exit if we didn't load a skeleton but have animations for one
    if (_skinnedMeshes.size() > 0)
        assert(_skeleton != NULL && "The skeleton wasn't impoted correctly."); // @todo rename all global variables by adding a prefex to them that identifies their use (rename skel here)
    
    // create the animation
    SLAnimation* result = new SLAnimation(animName, animDuration);

    SLbool isSkeletonAnim = false;
    for (SLuint i = 0; i < anim->mNumChannels; i++)
    {
        aiNodeAnim* channel = anim->mChannels[i];

        // find the node that is animated by this channel
        SLstring nodeName = channel->mNodeName.C_Str();
        SLNode* affectedNode = _sceneRoot->find<SLNode>(nodeName);
        SLuint handle = 0;
        SLbool isBoneNode = (affectedNode == NULL);

        // @todo: this is currently a work around but it can happen that we receive normal node animationtracks and joint animationtracks
        //        we don't allow node animation tracks in a skeleton animation, so we should split an animation in two seperate 
        //        animations if this happens. for now we just ignore node animation tracks if we already have skeleton tracks
        //        ofc this will crash if the first track is a node anim but its just temporary
        if (!isBoneNode && isSkeletonAnim) 
            continue;

        // is there a skeleton and is this animation channel not affecting a normal node?
        if (_skeletonRoot && !affectedNode)
        {
            isSkeletonAnim = true;
            SLJoint* affectedBone = _skeleton->getJoint(nodeName);
            if (affectedBone == NULL)
                break;
            handle = affectedBone->handle();
            // @todo warn if we find an animation with some node channels and some joint channels
            //       this shouldn't happen!

            /// @todo [high priority!] Fix the problem described below
            // What does this next line do?
            //   
            //   The testimportfile we used (Astroboy.dae) has the following properties:
            //      > It has joints in the skeleton that aren't animated by any channel.
            //      > The joints need a reset position of (0, 0, 0) to work properly 
            //          because the joint position is contained in a single keyframe for every joint
            //
            //      Since some of the joints don't have a channel that animates them, they also lack
            //      the joint position that the other joints get from their animation channel.
            //      So we need to set the initial state for all joints that have a channel
            //      to identity.
            //      All joints that arent in a channel will receive their local joint bind pose as
            //      reset position.
            //
            //      The problem stems from the design desicion to reset a whole skeleton before applying 
            //      animations to it. If we were to reset each joint just before applying a channel to it
            //      we wouldn't have this problem. But we coulnd't blend animations as easily.
            //
            //      @note (26-11-2014) about a week after this todo. This is the fault of the imported file
            //            the base animation must provide a keyframe for every joint in the skeleton or we will
            //            get problems when we try to interpolate with an other animation.
            SLMat4f prevOM = affectedBone->om();
            affectedBone->om(SLMat4f());
            affectedBone->setInitialState();
            affectedBone->om(prevOM);
        }
                            
        // log
        logMessage(LV_Normal, "\n  Channel %d %s", i, (isBoneNode) ? "(joint animation)\n" : "\n");
        logMessage(LV_Normal, "   Affected node: %s\n", channel->mNodeName.C_Str());
        logMessage(LV_Detailed, "   Num position keys: %d\n", channel->mNumPositionKeys);
        logMessage(LV_Detailed, "   Num rotation keys: %d\n", channel->mNumRotationKeys);
        logMessage(LV_Detailed, "   Num scaling keys: %d\n", channel->mNumScalingKeys);

        
        // joint animation channels should receive the correct node id, normal node animations just get 0
        SLNodeAnimationTrack* track = result->createNodeAnimationTrack(handle);

        
        // this is a node animation only, so we add a reference to the affected node to the track
        if (affectedNode && !isSkeletonAnim) {
            track->animationTarget(affectedNode);
        }

        KeyframeMap keyframes;

        // add position keys
        for (SLuint i = 0; i < channel->mNumPositionKeys; i++)
        {
            SLfloat time = (SLfloat)channel->mPositionKeys[i].mTime; // @todo test that we get the correct value back
            keyframes[time] = KeyframeData(&channel->mPositionKeys[i], NULL, NULL);
        }
        
        // add rotation keys
        for (SLuint i = 0; i < channel->mNumRotationKeys; i++)
        {
            SLfloat time = (SLfloat)channel->mRotationKeys[i].mTime; // @todo test that we get the correct value back

            if (keyframes.find(time) == keyframes.end())
                keyframes[time] = KeyframeData(NULL, &channel->mRotationKeys[i], NULL);
            else
            {
                // @todo this shouldn't abort but just throw an exception
                assert(keyframes[time].rotation == NULL && "There were two rotation keys assigned to the same timestamp.");
                keyframes[time].rotation = &channel->mRotationKeys[i];
            }
        }
        
        // add scaleing keys
        for (SLuint i = 0; i < channel->mNumScalingKeys; i++)
        {
            SLfloat time = (SLfloat)channel->mScalingKeys[i].mTime; // @todo test that we get the correct value back

            if (keyframes.find(time) == keyframes.end())
                keyframes[time] = KeyframeData(NULL, NULL, &channel->mScalingKeys[i]);
            else
            {
                // @todo this shouldn't abort but just throw an exception
                assert(keyframes[time].scaling == NULL && "There were two scaling keys assigned to the same timestamp.");
                keyframes[time].scaling = &channel->mScalingKeys[i];
            }
        }

             
        logMessage(LV_Normal, "   Found %d distinct keyframe timestamp(s).\n", keyframes.size());

        SLbool breakCondition = nodeName == "rig|Hips";

        KeyframeMap::iterator it = keyframes.begin();
        for (; it != keyframes.end(); it++)
        {
            SLTransformKeyframe* kf = track->createNodeKeyframe(it->first);         
            kf->translation(getTranslation(it->first, keyframes));
            kf->rotation(getRotation(it->first, keyframes));
            kf->scale(getScaling(it->first, keyframes));

            // log
            logMessage(LV_Detailed, "\n   Generating keyframe at time '%.2f'\n", it->first);
            logMessage(LV_Detailed, "    Translation: (%.2f, %.2f, %.2f) %s\n", kf->translation().x, kf->translation().y, kf->translation().z, (it->second.translation != NULL) ? "imported" : "generated");
            logMessage(LV_Detailed, "    Rotation: (%.2f, %.2f, %.2f, %.2f) %s\n", kf->rotation().x(), kf->rotation().y(), kf->rotation().z(), kf->rotation().w(), (it->second.rotation != NULL) ? "imported" : "generated");
            logMessage(LV_Detailed, "    Scale: (%.2f, %.2f, %.2f) %s\n", kf->scale().x, kf->scale().y, kf->scale().z, (it->second.scaling != NULL) ? "imported" : "generated");
        }
    }

    // add the animation to the skeleton or the importers animation list
    if (isSkeletonAnim)
        _skeleton->addAnimation(result);
    else
    {   
        SLScene::current->animManager().addNodeAnimation(result);
        _nodeAnimations.push_back(result);
    }
     

    return result;
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::aiNodeHasMesh returns true if the passed node or one of its
children has a mesh. aiNode can contain only transform or joint nodes without
any visuals.
*/
SLbool SLAssimpImporter::aiNodeHasMesh(aiNode* node)
{
    if (node->mNumMeshes > 0) return true;
    for(SLuint i = 0; i < node->mNumChildren; i++) 
        return aiNodeHasMesh(node->mChildren[i]);
    return false;
}
//-----------------------------------------------------------------------------
/*! 
SLAssimpImporter::checkFilePath tries to build the full absolut texture file path. 
Some file formats have absolut path stored, some have relative paths.
1st attempt: modelPath + aiTexFile
2nd attempt: aiTexFile
3rd attempt: modelPath + getFileName(aiTexFile)
If a model contains absolut path it is best to put all texture files beside the
model file in the same folder.
*/
SLstring SLAssimpImporter::checkFilePath(SLstring modelPath, SLstring aiTexFile)
{
    // Check path & file combination
    SLstring pathFile = modelPath + aiTexFile;
    if (SLFileSystem::fileExists(pathFile))
        return pathFile;

    // Check file alone
    if (SLFileSystem::fileExists(aiTexFile))
        return aiTexFile;

    // Check path & file combination
    pathFile = modelPath + SLUtils::getFileName(aiTexFile);
    if (SLFileSystem::fileExists(pathFile))
        return pathFile;

    SLstring msg = "SLAssimpImporter: Texture file not found: \n" + aiTexFile + 
                   "\non model path: " + modelPath + "\n";
    SL_WARN_MSG(msg.c_str());

    // Return path for texture not found image;
    return SLGLTexture::defaultPath + "TexNotFound.png";
}
//-----------------------------------------------------------------------------
