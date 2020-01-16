//#############################################################################
//  File:      SL/SLAssimpImporter.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <iomanip>
#include <Utils.h>

#include <SLAnimation.h>
#include <SLApplication.h>
#include <SLAssimpImporter.h>
#include <SLGLProgram.h>
#include <SLGLTexture.h>
#include <SLMaterial.h>
#include <SLScene.h>
#include <SLSkeleton.h>

// assimp is only included in the source file to not expose it to the rest of the framework
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

//-----------------------------------------------------------------------------
//! Temporary struct to hold keyframe data during assimp import.
struct SLImportKeyframe
{
    SLImportKeyframe()
      : translation(nullptr),
        rotation(nullptr),
        scaling(nullptr)
    {
    }

    SLImportKeyframe(aiVectorKey* trans, aiQuatKey* rot, aiVectorKey* scl)
    {
        translation = trans;
        rotation    = rot;
        scaling     = scl;
    }

    aiVectorKey* translation;
    aiQuatKey*   rotation;
    aiVectorKey* scaling;
};
typedef std::map<SLfloat, SLImportKeyframe> KeyframeMap;

//-----------------------------------------------------------------------------
/* Get the correct translation out of the keyframes map for a given time
this function interpolates linearly if no value is present in the map.
@note    this function does not wrap around to interpolate. if there is no
         translation key to the right of the passed in time then this function will take
         the last known value on the left!
*/
SLVec3f getTranslation(SLfloat time, const KeyframeMap& keyframes)
{
    KeyframeMap::const_iterator it = keyframes.find(time);
    aiVector3D                  result(0, 0, 0); // return 0 position of nothing was found

    // If the timestamp passed in doesnt exist then something in the loading of the kfs went wrong
    // @todo this should throw an exception and not kill the app
    assert(it != keyframes.end() && "A KeyframeMap was passed in with an illegal timestamp.");

    aiVectorKey* transKey = it->second.translation;

    // the timestamp has a valid translation value, just return the SL type
    if (transKey)
        result = transKey->mValue;
    else
    {
        aiVectorKey* frontKey = nullptr;
        aiVectorKey* backKey  = nullptr;

        // no translation value present, we must interpolate
        KeyframeMap::const_reverse_iterator revIt(it);

        // search to the right
        for (; it != keyframes.end(); it++)
        {
            if (it->second.translation != nullptr)
            {
                backKey = it->second.translation;
                break;
            }
        }

        // search to the left
        for (; revIt != keyframes.rend(); revIt++)
        {
            if (revIt->second.translation != nullptr)
            {
                frontKey = revIt->second.translation;
                break;
            }
        }

        if (frontKey && backKey)
        {
            SLfloat frontTime = revIt->first;
            SLfloat backTime  = it->first;
            SLfloat t         = (time - frontTime) / (backTime - frontTime);

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
//-----------------------------------------------------------------------------
/*! Get the correct scaling out of the keyframes map for a given time
    this function interpolates linearly if no value is present in the map.

    @note    this function does not wrap around to interpolate. if there is no 
             scaling key to the right of the passed in time then this function will take
             the last known value on the left!
*/
SLVec3f getScaling(SLfloat time, const KeyframeMap& keyframes)
{
    KeyframeMap::const_iterator it = keyframes.find(time);
    aiVector3D                  result(1, 1, 1); // return unit scale if no kf was found

    // If the timestamp passed in doesnt exist then something in the loading of the kfs went wrong
    // @todo this should throw an exception and not kill the app
    assert(it != keyframes.end() && "A KeyframeMap was passed in with an illegal timestamp.");

    aiVectorKey* scaleKey = it->second.scaling;

    // the timestamp has a valid translation value, just return the SL type
    if (scaleKey)
        result = scaleKey->mValue;
    else
    {
        aiVectorKey* frontKey = nullptr;
        aiVectorKey* backKey  = nullptr;

        // no translation value present, we must interpolate
        KeyframeMap::const_reverse_iterator revIt(it);

        // search to the right
        for (; it != keyframes.end(); it++)
        {
            if (it->second.rotation != nullptr)
            {
                backKey = it->second.scaling;
                break;
            }
        }

        // search to the left
        for (; revIt != keyframes.rend(); revIt++)
        {
            if (revIt->second.rotation != nullptr)
            {
                frontKey = revIt->second.scaling;
                break;
            }
        }

        if (frontKey && backKey)
        {
            SLfloat frontTime = revIt->first;
            SLfloat backTime  = it->first;
            SLfloat t         = (time - frontTime) / (backTime - frontTime);

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
//-----------------------------------------------------------------------------
/*! Get the correct rotation out of the keyframes map for a given time
    this function interpolates linearly if no value is present in the 

    @note    this function does not wrap around to interpolate. if there is no 
             rotation key to the right of the passed in time then this function will take
             the last known value on the left!
*/
SLQuat4f getRotation(SLfloat time, const KeyframeMap& keyframes)
{
    KeyframeMap::const_iterator it = keyframes.find(time);
    aiQuaternion                result(1, 0, 0, 0); // identity rotation

    // If the timesamp passed in doesnt exist then something in the loading of the kfs went wrong
    // @todo this should throw an exception and not kill the app
    assert(it != keyframes.end() && "A KeyframeMap was passed in with an illegal timestamp.");

    aiQuatKey* rotKey = it->second.rotation;

    // the timestamp has a valid translation value, just return the SL type
    if (rotKey)
        result = rotKey->mValue;
    else
    {
        aiQuatKey* frontKey = nullptr;
        aiQuatKey* backKey  = nullptr;

        // no translation value present, we must interpolate
        KeyframeMap::const_reverse_iterator revIt(it);

        // search to the right
        for (; it != keyframes.end(); it++)
        {
            if (it->second.rotation != nullptr)
            {
                backKey = it->second.rotation;
                break;
            }
        }

        // search to the left
        for (; revIt != keyframes.rend(); revIt++)
        {
            if (revIt->second.rotation != nullptr)
            {
                frontKey = revIt->second.rotation;
                break;
            }
        }

        if (frontKey && backKey)
        {
            SLfloat frontTime = revIt->first;
            SLfloat backTime  = it->first;
            SLfloat t         = (time - frontTime) / (backTime - frontTime);

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
added to the according vectors of SLScene for later deallocation. If an
override material is provided it will be assigned to all meshes and all
materials within the file are ignored.
*/
SLNode* SLAssimpImporter::load(SLstring    file,           //!< File with path or on default path
                               SLbool      loadMeshesOnly, //!< Only load nodes with meshes
                               SLMaterial* overrideMat,    //!< Override material
                               SLuint      flags)               //!< Import flags (see postprocess.h)
{
    // clear the intermediate data
    clear();

    // Check existance
    if (!Utils::fileExists(file))
    {
        file = defaultPath + file;
        if (!Utils::fileExists(file))
        {
            file = defaultPath + Utils::getFileName(file);
            if (!Utils::fileExists(file))
            {
                SLstring msg = "SLAssimpImporter: File not found: " + file + "\n";
                SL_WARN_MSG(msg.c_str());
                return nullptr;
            }
        }
    }

    // Import file with assimp importer
    Assimp::Importer ai;
    const aiScene*   scene = ai.ReadFile(file.c_str(), (SLuint)flags);
    if (!scene)
    {
        SLstring msg = "Failed to load file: " + file + "\n" + ai.GetErrorString() + "\n";
        SL_WARN_MSG(msg.c_str());
        return nullptr;
    }

    // initial scan of the scene
    performInitialScan(scene);

    // load skeleton
    loadSkeleton(nullptr, _skeletonRoot);

    // load materials
    SLstring    modelPath = Utils::getPath(file);
    SLVMaterial materials;
    if (!overrideMat)
    {
        for (SLint i = 0; i < (SLint)scene->mNumMaterials; i++)
            materials.push_back(loadMaterial(i, scene->mMaterials[i], modelPath));
    }

    // load meshes & set their material
    std::map<int, SLMesh*> meshMap; // map from the ai index to our mesh
    for (SLint i = 0; i < (SLint)scene->mNumMeshes; i++)
    {
        SLMesh* mesh = loadMesh(scene->mMeshes[i]);
        if (mesh != nullptr)
        {
            if (overrideMat)
                mesh->mat(overrideMat);
            else
                mesh->mat(materials[scene->mMeshes[i]->mMaterialIndex]);
            _meshes.push_back(mesh);
            meshMap[i] = mesh;
        }
        else
            SL_LOG("SLAsssimpImporter::load failed: %s\nin path: %s\n",
                   file.c_str(),
                   modelPath.c_str());
    }

    // load the scene nodes recursively
    _sceneRoot = loadNodesRec(nullptr, scene->mRootNode, meshMap, loadMeshesOnly);

    // load animations
    vector<SLAnimation*> animations;
    for (SLint i = 0; i < (SLint)scene->mNumAnimations; i++)
        animations.push_back(loadAnimation(scene->mAnimations[i]));

    logMessage(LV_minimal, "\n---------------------------\n\n");

    // Rename root node to the more meaningfull filename
    if (_sceneRoot)
        _sceneRoot->name(Utils::getFileName(file));

    return _sceneRoot;
}
//-----------------------------------------------------------------------------
//! Clears all helper containers
void SLAssimpImporter::clear()
{
    _nodeMap.clear();
    _jointOffsets.clear();
    _skeletonRoot = nullptr;
    _skeleton     = nullptr;
    _skinnedMeshes.clear();
}
//-----------------------------------------------------------------------------
//! Return an aiNode ptr if name exists, or null if it doesn't
aiNode* SLAssimpImporter::getNodeByName(const SLstring& name)
{
    if (_nodeMap.find(name) != _nodeMap.end())
        return _nodeMap[name];

    return nullptr;
}
//-----------------------------------------------------------------------------
//! Returns an aiBone ptr if name exists, or null if it doesn't
const SLMat4f SLAssimpImporter::getOffsetMat(const SLstring& name)
{
    if (_jointOffsets.find(name) != _jointOffsets.end())
        return _jointOffsets[name];

    return SLMat4f();
}
//-----------------------------------------------------------------------------
//! Populates nameToNode, nameToBone, jointGroups, skinnedMeshes
void SLAssimpImporter::performInitialScan(const aiScene* scene)
{
    // populate the _nameToNode map and print the assimp structure on detailed log verbosity.
    logMessage(LV_detailed, "[Assimp scene]\n");
    logMessage(LV_detailed, "  Cameras: %d\n", scene->mNumCameras);
    logMessage(LV_detailed, "  Lights: %d\n", scene->mNumLights);
    logMessage(LV_detailed, "  Meshes: %d\n", scene->mNumMeshes);
    logMessage(LV_detailed, "  Materials: %d\n", scene->mNumMaterials);
    logMessage(LV_detailed, "  Textures: %d\n", scene->mNumTextures);
    logMessage(LV_detailed, "  Animations: %d\n", scene->mNumAnimations);

    logMessage(LV_detailed, "---------------------------------------------\n");
    logMessage(LV_detailed, "  Node node tree: \n");
    findNodes(scene->mRootNode, "  ", true);

    logMessage(LV_detailed, "---------------------------------------------\n");
    logMessage(LV_detailed, "   Searching for skinned meshes and scanning joint names.\n");

    findJoints(scene);
    findSkeletonRoot();
}
//-----------------------------------------------------------------------------
//! Scans the assimp scene graph structure and populates nameToNode
void SLAssimpImporter::findNodes(aiNode* node, SLstring padding, SLbool lastChild)
{
    SLstring name = node->mName.C_Str();
    /*
    /// @todo we can't allow for duplicate node names, ever at the moment. The 'solution' below
    ///       only hides the problem and moves it to a different part.
    // rename duplicate node names
    SLstring renamedString;
    if (_nodeMap.find(name) != _nodeMap.end())
    {
        SLint index = 0;
        std::ostringstream ss;
        SLstring lastMatch = name;
        while (_nodeMap.find(lastMatch) != _nodeMap.end()) 
        {
            ss.str(SLstring());
            ss.clear();
            ss << name << "_" << std::setw( 2 ) << std::setfill( '0' ) << index;
            lastMatch = ss.str();
            index++;
        }
        ss.str(SLstring());
        ss.clear();
        ss << "(renamed from '" << name << "')";
        renamedString = ss.str();
        name = lastMatch;
    }*/

    // this should not happen
    assert(_nodeMap.find(name) == _nodeMap.end() && "Duplicated node name found!");
    _nodeMap[name] = node;

    //logMessage(LV_Detailed, "%s   |\n", padding.c_str());
    logMessage(LV_detailed,
               "%s  |-[%s]   (%d children, %d meshes)\n",
               padding.c_str(),
               name.c_str(),
               node->mNumChildren,
               node->mNumMeshes);

    if (lastChild)
        padding += "   ";
    else
        padding += "  |";

    for (SLuint i = 0; i < node->mNumChildren; i++)
    {
        findNodes(node->mChildren[i], padding, (i == node->mNumChildren - 1));
    }
}
//-----------------------------------------------------------------------------
/*! Scans all meshes in the assimp scene and populates nameToBone and
jointGroups
*/
void SLAssimpImporter::findJoints(const aiScene* scene)
{
    for (SLuint i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];
        if (!mesh->HasBones())
            continue;

        logMessage(LV_normal,
                   "   Mesh '%s' contains %d joints.\n",
                   mesh->mName.C_Str(),
                   mesh->mNumBones);

        for (SLuint j = 0; j < mesh->mNumBones; j++)
        {
            SLstring                              name = mesh->mBones[j]->mName.C_Str();
            std::map<SLstring, SLMat4f>::iterator it   = _jointOffsets.find(name);
            if (it != _jointOffsets.end())
                continue;

            // add the offset matrix to our offset matrix map
            SLMat4f offsetMat;
            memcpy(&offsetMat, &mesh->mBones[j]->mOffsetMatrix, sizeof(SLMat4f));
            offsetMat.transpose();
            _jointOffsets[name] = offsetMat;

            logMessage(LV_detailed, "     Bone '%s' found.\n", name.c_str());
        }
    }
}
//-----------------------------------------------------------------------------
/*! Finds the common ancestor for each remaining group in jointGroups,
these are our final skeleton roots
*/
void SLAssimpImporter::findSkeletonRoot()
{
    _skeletonRoot = nullptr;
    // early out if we don't have any joint bindings
    if (_jointOffsets.empty()) return;

    vector<SLVaiNode> ancestorList(_jointOffsets.size());
    SLint             minDepth = INT_MAX;
    SLuint            index    = 0;

    logMessage(LV_detailed, "Building joint ancestor lists.\n");

    auto it = _jointOffsets.begin();
    for (; it != _jointOffsets.end(); it++, index++)
    {
        aiNode*    node = getNodeByName(it->first);
        SLVaiNode& list = ancestorList[index];

        while (node)
        {
            list.insert(list.begin(), node);
            node = node->mParent;
        }

        // log the gathered ancestor list if on diagnostic
        if (LV_diagnostic)
        {
            logMessage(LV_diagnostic,
                       "   '%s' ancestor list: ",
                       it->first.c_str());

            for (SLuint i = 0; i < list.size(); i++)
                logMessage(LV_diagnostic,
                           "'%s' ",
                           list[i]->mName.C_Str());

            logMessage(LV_diagnostic, "\n");
        }
        else
            logMessage(LV_detailed,
                       "   '%s' lies at a depth of %d\n",
                       it->first.c_str(),
                       list.size());

        minDepth = std::min(minDepth, (SLint)list.size());
    }

    logMessage(LV_detailed,
               "Bone ancestor lists completed, min depth: %d\n",
               minDepth);

    logMessage(LV_detailed,
               "Searching ancestor lists for common ancestor.\n");

    // now we have a ancestor list for each joint node beginning with the root node
    for (SLuint i = 0; i < (SLuint)minDepth; i++)
    {
        SLbool  failed    = false;
        aiNode* lastMatch = ancestorList[0][i];
        for (SLuint j = 1; j < ancestorList.size(); j++)
        {
            if (ancestorList[j][i] != lastMatch)
                failed = true;

            lastMatch = ancestorList[j][i];
        }

        // all ancestors matched
        if (!failed)
        {
            _skeletonRoot = lastMatch;
            logMessage(LV_detailed,
                       "Found matching ancestor '%s'.\n",
                       _skeletonRoot->mName.C_Str());
        }
        else
        {
            break;
        }
    }

    // seems like the above can be wrong, we should just select the common
    // ancestor that is one below the assimps root.
    // @todo fix this function up and make sure there exists a second element
    if (!_skeletonRoot)
        _skeletonRoot = ancestorList[0][1];

    logMessage(LV_normal,
               "Determined '%s' to be the skeleton's root node.\n",
               _skeletonRoot->mName.C_Str());
}
//-----------------------------------------------------------------------------
//! Loads the skeleton
void SLAssimpImporter::loadSkeleton(SLJoint* parent, aiNode* node)
{
    if (!node)
        return;

    SLJoint* joint;
    SLstring name = node->mName.C_Str();

    if (!parent)
    {
        logMessage(LV_normal, "Loading skeleton skeleton.\n");
        _skeleton   = new SLSkeleton;
        _jointIndex = 0;

        joint = _skeleton->createJoint(name, _jointIndex++);
        _skeleton->rootJoint(joint);
    }
    else
    {
        joint = parent->createChild(name, _jointIndex++);
    }

    joint->offsetMat(getOffsetMat(name));

    // set the initial state for the joints (in case we render the model
    // without playing its animation) an other possibility is to set the joints
    // to the inverse offset matrix so that the model remains in its bind pose
    // some files will report the node transformation as the animation state
    // transformation that the model had when exporting (in case of our astroboy
    // its in the middle of the animation.
    // It might be more desirable to have ZERO joint transformations in the initial
    // pose to be able to see the model without any joint modifications applied
    // exported state

    // set the current node transform as the initial state
    /*
    SLMat4f om;
    memcpy(&om, &node->mTransformation, sizeof(SLMat4f));
    om.transpose();
    joint->om(om);
    joint->setInitialState();
    */
    // set the binding pose as initial state
    SLMat4f om;
    om = joint->offsetMat().inverted();
    if (parent)
        om = parent->updateAndGetWM().inverted() * om;
    joint->om(om);
    joint->setInitialState();

    for (SLuint i = 0; i < node->mNumChildren; i++)
        loadSkeleton(joint, node->mChildren[i]);
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadMaterial loads the AssImp material an returns the SLMaterial.
The materials and textures are added to the SLScene material and texture 
vectors.
*/
SLMaterial* SLAssimpImporter::loadMaterial(SLint       index,
                                           aiMaterial* material,
                                           SLstring    modelPath)
{
    // Get the materials name
    aiString matName;
    material->Get(AI_MATKEY_NAME, matName);
    SLstring name = matName.data;
    if (name.empty()) name = "Import Material";

    // Create SLMaterial instance. It is also added to the SLScene::_materials vector
    SLMaterial* mat = new SLMaterial(name.c_str());

    // set the texture types to import into our material
    const SLint   textureCount = 5;
    aiTextureType textureTypes[textureCount];
    textureTypes[0] = aiTextureType_DIFFUSE;
    textureTypes[1] = aiTextureType_NORMALS;
    textureTypes[2] = aiTextureType_SPECULAR;
    textureTypes[3] = aiTextureType_HEIGHT;
    textureTypes[4] = aiTextureType_OPACITY; // Texture with alpha channel

    // load all the textures for this material and add it to the material vector
    for (SLint i = 0; i < textureCount; ++i)
    {
        if (material->GetTextureCount(textureTypes[i]) > 0)
        {
            aiString aipath;
            material->GetTexture(textureTypes[i], 0, &aipath, nullptr, nullptr, nullptr, nullptr, nullptr);
            SLTextureType texType = textureTypes[i] == aiTextureType_DIFFUSE
                                      ? TT_color
                                      : textureTypes[i] == aiTextureType_NORMALS
                                          ? TT_normal
                                          : textureTypes[i] == aiTextureType_SPECULAR
                                              ? TT_gloss
                                              : textureTypes[i] == aiTextureType_HEIGHT
                                                  ? TT_height
                                                  : textureTypes[i] == aiTextureType_OPACITY
                                                      ? TT_color
                                                      : TT_unknown;
            SLstring texFile = checkFilePath(modelPath, aipath.data);

            // Only color texture are loaded so far
            // For normal maps we have to adjust first the normal and tangent generation
            if (texType == TT_color)
            {
                SLGLTexture* tex = loadTexture(texFile, texType);
                mat->textures().push_back(tex);
            }
        }
    }

    // get color data
    aiColor3D ambient, diffuse, specular, emissive;
    SLfloat   shininess, refracti, reflectivity, opacity;
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
    mat->emissive(SLCol4f(emissive.r, emissive.g, emissive.b));
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
SLGLTexture* SLAssimpImporter::loadTexture(SLstring&     textureFile,
                                           SLTextureType texType)
{
    SLVGLTexture& sceneTex = SLApplication::scene->textures();

    // return if a texture with the same file allready exists
    for (SLuint i = 0; i < sceneTex.size(); ++i)
        if (sceneTex[i]->name() == textureFile)
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
triangle face indices. Normals & tangents are not loaded. They are calculated
in SLMesh.
*/
SLMesh* SLAssimpImporter::loadMesh(aiMesh* mesh)
{
    // Count first the NO. of triangles in the mesh
    SLuint numPoints    = 0;
    SLuint numLines     = 0;
    SLuint numTriangles = 0;
    SLuint numPolygons  = 0;

    for (unsigned int i = 0; i < mesh->mNumFaces; ++i)
    {
        if (mesh->mFaces[i].mNumIndices == 1) numPoints++;
        if (mesh->mFaces[i].mNumIndices == 2) numLines++;
        if (mesh->mFaces[i].mNumIndices == 3) numTriangles++;
        if (mesh->mFaces[i].mNumIndices > 3) numPolygons++;
    }

    // A mesh can contain either point, lines or triangles
    if ((numTriangles && (numLines || numPoints)) ||
        (numLines && (numTriangles || numPoints)) ||
        (numPoints && (numLines || numTriangles)))
    {
        SL_LOG("SLAssimpImporter::loadMesh:  Mesh contains multiple primitive types: %s, Lines: %d, Points: %d\n",
               mesh->mName.C_Str(),
               numLines,
               numPoints);

        // Prioritize triangles over lines over points
        if (numTriangles && numLines) numLines = 0;
        if (numTriangles && numPoints) numPoints = 0;
        if (numLines && numPoints) numPoints = 0;
    }

    if (numPolygons > 0)
    {
        SL_LOG("SLAssimpImporter::loadMesh:  Mesh contains polygons: %s\n",
               mesh->mName.C_Str());
        return nullptr;
    }

    // We only load meshes that contain triangles or lines
    if (mesh->mNumVertices == 0)
    {
        SL_LOG("SLAssimpImporter::loadMesh:  Mesh has no vertices: %s\n",
               mesh->mName.C_Str());
        return nullptr;
    }

    // We only load meshes that contain triangles or lines
    if (numTriangles == 0 && numLines == 0 && numPoints == 0)
    {
        SL_LOG("SLAssimpImporter::loadMesh:  Mesh has has no triangles nor lines nor points: %s\n",
               mesh->mName.C_Str());
        return nullptr;
    }

    // create a new mesh.
    // The mesh pointer is added automatically to the SLScene::meshes vector.
    SLstring name = mesh->mName.data;
    SLMesh*  m    = new SLMesh(name.empty() ? "Imported Mesh" : name);

    // Set primitive type
    if (numTriangles) m->primitive(SLGLPrimitiveType::PT_triangles);
    if (numLines) m->primitive(SLGLPrimitiveType::PT_lines);
    if (numPoints) m->primitive(SLGLPrimitiveType::PT_points);

    // create position & normal vector
    m->P.clear();
    m->P.resize(mesh->mNumVertices);

    // create normal vector for triangle primitive types
    if (mesh->HasNormals() && numTriangles)
    {
        m->N.clear();
        m->N.resize(m->P.size());
    }

    // allocate texCoord vector if needed
    if (mesh->HasTextureCoords(0) && numTriangles)
    {
        m->Tc.clear();
        m->Tc.resize(m->P.size());
    }

    // copy vertex positions & texCoord
    for (SLuint i = 0; i < m->P.size(); ++i)
    {
        m->P[i].set(mesh->mVertices[i].x,
                    mesh->mVertices[i].y,
                    mesh->mVertices[i].z);
        if (m->N.size())
            m->N[i].set(mesh->mNormals[i].x,
                        mesh->mNormals[i].y,
                        mesh->mNormals[i].z);
        if (m->Tc.size())
            m->Tc[i].set(mesh->mTextureCoords[0][i].x,
                         mesh->mTextureCoords[0][i].y);
    }

    // create primitive index vector
    SLuint j = 0;
    if (m->P.size() < 65536)
    {
        m->I16.clear();
        if (numTriangles)
        {
            m->I16.resize(numTriangles * 3);
            for (SLuint i = 0; i < mesh->mNumFaces; ++i)
            {
                if (mesh->mFaces[i].mNumIndices == 3)
                {
                    m->I16[j++] = (SLushort)mesh->mFaces[i].mIndices[0];
                    m->I16[j++] = (SLushort)mesh->mFaces[i].mIndices[1];
                    m->I16[j++] = (SLushort)mesh->mFaces[i].mIndices[2];
                }
            }
        }
        else if (numLines)
        {
            m->I16.resize(numLines * 2);
            for (SLuint i = 0; i < mesh->mNumFaces; ++i)
            {
                if (mesh->mFaces[i].mNumIndices == 2)
                {
                    m->I16[j++] = (SLushort)mesh->mFaces[i].mIndices[0];
                    m->I16[j++] = (SLushort)mesh->mFaces[i].mIndices[1];
                }
            }
        }
        else if (numPoints)
        {
            m->I16.resize(numPoints);
            for (SLuint i = 0; i < mesh->mNumFaces; ++i)
            {
                if (mesh->mFaces[i].mNumIndices == 1)
                    m->I16[j++] = (SLushort)mesh->mFaces[i].mIndices[0];
            }
        }

        // check for invalid indices
        for (auto i : m->I16)
            assert(i < m->P.size() && "SLAssimpImporter::loadMesh: Invalid Index");
    }
    else
    {
        m->I32.clear();
        if (numTriangles)
        {
            m->I32.resize(numTriangles * 3);
            for (SLuint i = 0; i < mesh->mNumFaces; ++i)
            {
                if (mesh->mFaces[i].mNumIndices == 3)
                {
                    m->I32[j++] = mesh->mFaces[i].mIndices[0];
                    m->I32[j++] = mesh->mFaces[i].mIndices[1];
                    m->I32[j++] = mesh->mFaces[i].mIndices[2];
                }
            }
        }
        else if (numLines)
        {
            m->I32.resize(numLines * 2);
            for (SLuint i = 0; i < mesh->mNumFaces; ++i)
            {
                if (mesh->mFaces[i].mNumIndices == 2)
                {
                    m->I32[j++] = mesh->mFaces[i].mIndices[0];
                    m->I32[j++] = mesh->mFaces[i].mIndices[1];
                }
            }
        }
        else if (numPoints)
        {
            m->I32.resize(numPoints * 1);
            for (SLuint i = 0; i < mesh->mNumFaces; ++i)
            {
                if (mesh->mFaces[i].mNumIndices == 1)
                    m->I32[j++] = mesh->mFaces[i].mIndices[0];
            }
        }

        // check for invalid indices
        for (auto i : m->I32)
            assert(i < m->P.size() && "SLAssimpImporter::loadMesh: Invalid Index");
    }

    if (!mesh->HasNormals() && numTriangles)
        m->calcNormals();

    // load joints
    if (mesh->HasBones())
    {
        _skinnedMeshes.push_back(m);
        m->skeleton(_skeleton);

        m->Ji.resize(m->P.size());
        m->Jw.resize(m->P.size());

        for (SLuint i = 0; i < mesh->mNumBones; i++)
        {
            aiBone*  joint   = mesh->mBones[i];
            SLJoint* slJoint = _skeleton->getJoint(joint->mName.C_Str());

            // @todo On OSX it happens from time to time that slJoint is nullptr
            if (slJoint)
            {
                for (SLuint j = 0; j < joint->mNumWeights; j++)
                {
                    // add the weight
                    SLuint  vertId = joint->mWeights[j].mVertexId;
                    SLfloat weight = joint->mWeights[j].mWeight;

                    m->Ji[vertId].push_back((SLuchar)slJoint->id());
                    m->Jw[vertId].push_back(weight);

                    // check if the bones max radius changed
                    // @todo this is very specific to this loaded mesh,
                    //       when we add a skeleton instances class this radius
                    //       calculation has to be done on the instance!
                    slJoint->calcMaxRadius(SLVec3f(mesh->mVertices[vertId].x,
                                                   mesh->mVertices[vertId].y,
                                                   mesh->mVertices[vertId].z));
                }
            }
            else
            {
                SL_LOG("Failed to load joint of skeleton in SLAssimpImporter::loadMesh: %s\n",
                       joint->mName.C_Str());
                //return nullptr;
            }
        }
    }

    return m;
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadNodesRec loads the scene graph node tree recursively.
*/
SLNode* SLAssimpImporter::loadNodesRec(
  SLNode*    curNode,    //!< Pointer to the current node. Pass nullptr for root node
  aiNode*    node,       //!< The according assimp node. Pass nullptr for root node
  SLMeshMap& meshes,     //!< Reference to the meshes vector
  SLbool     loadMeshesOnly) //!< Only load nodes with meshes
{
    // we're at the root
    if (!curNode)
        curNode = new SLNode(node->mName.data);

    // load local transform
    aiMatrix4x4* M = &node->mTransformation;

    // clang-format off
    SLMat4f      SLM(M->a1, M->a2, M->a3, M->a4,
                     M->b1, M->b2, M->b3, M->b4,
                     M->c1, M->c2, M->c3, M->c4,
                     M->d1, M->d2, M->d3, M->d4);
    // clang-format on

    curNode->om(SLM);

    // add the meshes
    for (SLuint i = 0; i < node->mNumMeshes; ++i)
    {
        // Only add meshes that were added to the meshMap (triangle meshes)
        if (meshes.count((SLint)node->mMeshes[i]))
            curNode->addMesh(meshes[(SLint)node->mMeshes[i]]);
    }

    // load children recursively
    for (SLuint i = 0; i < node->mNumChildren; i++)
    {
        // skip the skeleton
        if (node->mChildren[i] == _skeletonRoot)
            continue;

        // only add subtrees that contain a mesh in one of their nodes
        if (!loadMeshesOnly || aiNodeHasMesh(node->mChildren[i]))
        {
            SLNode* child = new SLNode(node->mChildren[i]->mName.data);
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
SLAnimation* SLAssimpImporter::loadAnimation(aiAnimation* anim)
{
    ostringstream oss;
    oss << "unnamed_anim_" << SLApplication::scene->animManager().allAnimNames().size();
    SLstring animName        = oss.str();
    SLfloat  animTicksPerSec = (anim->mTicksPerSecond < 0.0001f)
                                ? 30.0f
                                : (SLfloat)anim->mTicksPerSecond;
    SLfloat animDuration = (SLfloat)anim->mDuration / animTicksPerSec;

    if (anim->mName.length > 0)
        animName = anim->mName.C_Str();

    // log
    logMessage(LV_minimal, "\nLoading animation %s\n", animName.c_str());
    logMessage(LV_normal, " Duration(seconds): %f \n", animDuration);
    logMessage(LV_normal, " Duration(ticks): %f \n", anim->mDuration);
    logMessage(LV_normal, " Ticks per second: %f \n", animTicksPerSec);
    logMessage(LV_normal, " Num channels: %d\n", anim->mNumChannels);

    // exit if we didn't load a skeleton but have animations for one
    if (!_skinnedMeshes.empty())
        assert(_skeleton != nullptr && "The skeleton wasn't impoted correctly.");

    // create the animation
    SLAnimation* result;
    if (_skeleton)
        result = _skeleton->createAnimation(animName, animDuration);
    else
    {
        result = SLApplication::scene->animManager().createNodeAnimation(animName, animDuration);
        _nodeAnimations.push_back(result);
    }

    SLbool isSkeletonAnim = false;
    for (SLuint i = 0; i < anim->mNumChannels; i++)
    {
        aiNodeAnim* channel = anim->mChannels[i];

        // find the node that is animated by this channel
        SLstring nodeName     = channel->mNodeName.C_Str();
        SLNode*  affectedNode = _sceneRoot->find<SLNode>(nodeName);
        SLuint   id           = 0;
        SLbool   isJointNode  = (affectedNode == nullptr);

        // @todo: this is currently a work around but it can happen that we receive normal node animationtracks and joint animationtracks
        //        we don't allow node animation tracks in a skeleton animation, so we should split an animation in two seperate
        //        animations if this happens. for now we just ignore node animation tracks if we already have joint tracks
        //        ofc this will crash if the first track is a node anim but its just temporary
        if (!isJointNode && isSkeletonAnim)
            continue;

        // is there a skeleton and is this animation channel not affecting a normal node?
        if (_skeletonRoot && !affectedNode)
        {
            isSkeletonAnim         = true;
            SLJoint* affectedJoint = _skeleton->getJoint(nodeName);
            if (affectedJoint == nullptr)
                break;

            id = affectedJoint->id();
            // @todo warn if we find an animation with some node channels and some joint channels
            //       this shouldn't happen!

            /// @todo [high priority!] Address the problem of some bones not containing an animation channel
            ///         when importing. Current workaround is to set their reset position to their bind pose.
            ///         This will however fail if we have multiple animations affecting a single model and fading
            ///         some of them out or in. This will require us to provide animations that have a channel
            ///         for all bones even if they're just positional.
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
            SLMat4f prevOM = affectedJoint->om();
            affectedJoint->om(SLMat4f());
            affectedJoint->setInitialState();
            affectedJoint->om(prevOM);
        }

        // log
        logMessage(LV_normal, "\n  Channel %d %s", i, (isJointNode) ? "(joint animation)\n" : "\n");
        logMessage(LV_normal, "   Affected node: %s\n", channel->mNodeName.C_Str());
        logMessage(LV_detailed, "   Num position keys: %d\n", channel->mNumPositionKeys);
        logMessage(LV_detailed, "   Num rotation keys: %d\n", channel->mNumRotationKeys);
        logMessage(LV_detailed, "   Num scaling keys: %d\n", channel->mNumScalingKeys);

        // joint animation channels should receive the correct node id, normal node animations just get 0
        SLNodeAnimTrack* track = result->createNodeAnimationTrack(id);

        // this is a node animation only, so we add a reference to the affected node to the track
        if (affectedNode && !isSkeletonAnim)
        {
            track->animatedNode(affectedNode);
        }

        KeyframeMap keyframes;

        // add position keys
        for (SLuint i = 0; i < channel->mNumPositionKeys; i++)
        {
            SLfloat time    = (SLfloat)channel->mPositionKeys[i].mTime;
            keyframes[time] = SLImportKeyframe(&channel->mPositionKeys[i], nullptr, nullptr);
        }

        // add rotation keys
        for (SLuint i = 0; i < channel->mNumRotationKeys; i++)
        {
            SLfloat time = (SLfloat)channel->mRotationKeys[i].mTime;

            if (keyframes.find(time) == keyframes.end())
                keyframes[time] = SLImportKeyframe(nullptr, &channel->mRotationKeys[i], nullptr);
            else
            {
                // @todo this shouldn't abort but just throw an exception
                assert(keyframes[time].rotation == nullptr && "There were two rotation keys assigned to the same timestamp.");
                keyframes[time].rotation = &channel->mRotationKeys[i];
            }
        }

        // add scaleing keys
        for (SLuint i = 0; i < channel->mNumScalingKeys; i++)
        {
            SLfloat time = (SLfloat)channel->mScalingKeys[i].mTime;

            if (keyframes.find(time) == keyframes.end())
                keyframes[time] = SLImportKeyframe(nullptr, nullptr, &channel->mScalingKeys[i]);
            else
            {
                // @todo this shouldn't abort but just throw an exception
                assert(keyframes[time].scaling == nullptr && "There were two scaling keys assigned to the same timestamp.");
                keyframes[time].scaling = &channel->mScalingKeys[i];
            }
        }

        logMessage(LV_normal, "   Found %d distinct keyframe timestamp(s).\n", keyframes.size());

        for (auto it : keyframes)
        {
            SLTransformKeyframe* kf = track->createNodeKeyframe(it.first);
            kf->translation(getTranslation(it.first, keyframes));
            kf->rotation(getRotation(it.first, keyframes));
            kf->scale(getScaling(it.first, keyframes));

            // log
            logMessage(LV_detailed,
                       "\n   Generating keyframe at time '%.2f'\n",
                       it.first);
            logMessage(LV_detailed,
                       "    Translation: (%.2f, %.2f, %.2f) %s\n",
                       kf->translation().x,
                       kf->translation().y,
                       kf->translation().z,
                       (it.second.translation != nullptr) ? "imported" : "generated");
            logMessage(LV_detailed,
                       "    Rotation: (%.2f, %.2f, %.2f, %.2f) %s\n",
                       kf->rotation().x(),
                       kf->rotation().y(),
                       kf->rotation().z(),
                       kf->rotation().w(),
                       (it.second.rotation != nullptr) ? "imported" : "generated");
            logMessage(LV_detailed,
                       "    Scale: (%.2f, %.2f, %.2f) %s\n",
                       kf->scale().x,
                       kf->scale().y,
                       kf->scale().z,
                       (it.second.scaling != nullptr) ? "imported" : "generated");
        }
    }

    return result;
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::aiNodeHasMesh returns true if the passed node or one of its
children has a mesh. aiNode can contain only transform or joint nodes without
any visuals.

@todo   this function doesn't look well optimized. It's currently used if the option to
        only load nodes containing meshes somewhere in their heirarchy is enabled.
        This means we call it on ancestor nodes first. This also means that we will 
        redundantly traverse the same exact nodes multiple times. This isn't a pressing
        issue at the moment but should be tackled when this importer is being optimized
*/
SLbool SLAssimpImporter::aiNodeHasMesh(aiNode* node)
{
    if (node->mNumMeshes > 0) return true;

    for (SLuint i = 0; i < node->mNumChildren; i++)
        if (node->mChildren[i]->mNumMeshes > 0)
            return aiNodeHasMesh(node->mChildren[i]);
    return false;
}
//-----------------------------------------------------------------------------
/*! 
SLAssimpImporter::checkFilePath tries to build the full absolut texture file path. 
Some file formats have absolute path stored, some have relative paths.
1st attempt: modelPath + aiTexFile
2nd attempt: aiTexFile
3rd attempt: modelPath + getFileName(aiTexFile)
If a model contains absolute path it is best to put all texture files beside the
model file in the same folder.
*/
SLstring SLAssimpImporter::checkFilePath(SLstring modelPath, SLstring aiTexFile)
{
    // Check path & file combination
    SLstring pathFile = modelPath + aiTexFile;
    if (Utils::fileExists(pathFile))
        return pathFile;

    // Check file alone
    if (Utils::fileExists(aiTexFile))
        return aiTexFile;

    // Check path & file combination
    pathFile = modelPath + Utils::getFileName(aiTexFile);
    if (Utils::fileExists(pathFile))
        return pathFile;

    SLstring msg = "SLAssimpImporter: Texture file not found: \n" + aiTexFile +
                   "\non model path: " + modelPath + "\n";
    SL_WARN_MSG(msg.c_str());

    // Return path for texture not found image;
    return SLGLTexture::defaultPath + "TexNotFound.png";
}
//-----------------------------------------------------------------------------
