//#############################################################################
//  File:      sl/SLAssimpImporter.cpp
//  Authors:   Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_BUILD_WITH_ASSIMP

#    include <iomanip>
#    include <Utils.h>

#    include <SLAnimation.h>
#    include <SLAssimpImporter.h>
#    include <SLGLTexture.h>
#    include <SLMaterial.h>
#    include <SLAnimSkeleton.h>
#    include <SLAssetManager.h>
#    include <SLAnimManager.h>
#    include <Profiler.h>
#    include <SLAssimpProgressHandler.h>
#    include <SLAssimpIOSystem.h>

// assimp is only included in the source file to not expose it to the rest of the framework
#    include <assimp/Importer.hpp>
#    include <assimp/scene.h>
#    include <assimp/pbrmaterial.h>

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
SLNode* SLAssimpImporter::load(SLAnimManager&     aniMan,                 //!< Reference to the animation manager
                               SLAssetManager*    assetMgr,               //!< Pointer to the asset manager
                               SLstring           pathAndFile,            //!< File with path or on default path
                               SLstring           texturePath,            //!< Path to the texture images
                               SLSkybox*          skybox,                 //!< Pointer to the skybox
                               SLbool             deleteTexImgAfterBuild, //!< Default = false
                               SLbool             loadMeshesOnly,         //!< Default = true
                               SLMaterial*        overrideMat,            //!< Override material
                               float              ambientFactor,          //!< if ambientFactor > 0 ambient = diffuse * AmbientFactor
                               SLbool             forceCookTorranceRM,    //!< Forces Cook-Torrance reflection model
                               SLProgressHandler* progressHandler,        //!< Pointer to progress handler
                               SLuint             flags                   //!< Import flags (see postprocess.h)
)
{
    PROFILE_FUNCTION();

    // clear the intermediate data
    clear();

    // Check existence
    if (!SLFileStorage::exists(pathAndFile, IOK_shader))
    {
        SLstring msg = "SLAssimpImporter: File not found: " + pathAndFile + "\n";
        SL_EXIT_MSG(msg.c_str());
        return nullptr;
    }

    // Import file with assimp importer
    Assimp::Importer ai;

    // Set progress handler
    if (progressHandler)
        ai.SetProgressHandler((Assimp::ProgressHandler*)progressHandler);

    ///////////////////////////////////////////////////////////////////////
    ai.SetIOHandler(new SLAssimpIOSystem());
    const aiScene* scene = ai.ReadFile(pathAndFile, (SLuint)flags);
    ///////////////////////////////////////////////////////////////////////

    if (!scene)
    {
        SLstring msg = "Failed to load file: " + pathAndFile + "\n" + ai.GetErrorString() + "\n";
        SL_WARN_MSG(msg.c_str());
        return nullptr;
    }

    // initial scan of the scene
    performInitialScan(scene);

    // load skeleton
    loadSkeleton(aniMan, nullptr, _skeletonRoot);

    // load materials
    SLstring    modelPath = Utils::getPath(pathAndFile);
    SLVMaterial materials;
    if (!overrideMat)
    {
        for (SLint i = 0; i < (SLint)scene->mNumMaterials; i++)
            materials.push_back(loadMaterial(assetMgr,
                                             i,
                                             scene->mMaterials[i],
                                             modelPath,
                                             texturePath,
                                             skybox,
                                             ambientFactor,
                                             forceCookTorranceRM,
                                             deleteTexImgAfterBuild));
    }

    // load meshes & set their material
    std::map<int, SLMesh*> meshMap; // map from the ai index to our mesh
    for (SLint i = 0; i < (SLint)scene->mNumMeshes; i++)
    {
        SLMesh* mesh = loadMesh(assetMgr, scene->mMeshes[i]);
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
            SL_LOG("SLAsssimpImporter::load failed: %s\nin path: %s",
                   pathAndFile.c_str(),
                   modelPath.c_str());
    }

    // load the scene nodes recursively
    _sceneRoot = loadNodesRec(nullptr, scene->mRootNode, meshMap, loadMeshesOnly);

    // load animations
    vector<SLAnimation*> animations;
    for (SLint i = 0; i < (SLint)scene->mNumAnimations; i++)
        animations.push_back(loadAnimation(aniMan, scene->mAnimations[i]));

    logMessage(LV_minimal, "\n---------------------------\n\n");

    // Rename root node to the more meaningfull filename
    if (_sceneRoot)
        _sceneRoot->name(Utils::getFileName(pathAndFile));

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
SLMat4f SLAssimpImporter::getOffsetMat(const SLstring& name)
{
    if (_jointOffsets.find(name) != _jointOffsets.end())
        return _jointOffsets[name];

    return SLMat4f();
}
//-----------------------------------------------------------------------------
//! Populates nameToNode, nameToBone, jointGroups, skinnedMeshes
void SLAssimpImporter::performInitialScan(const aiScene* scene)
{
    PROFILE_FUNCTION();

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

    // logMessage(LV_Detailed, "%s   |\n", padding.c_str());
    logMessage(LV_detailed,
               "%s  |-[%s]   (%d children, %d meshes)",
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

            for (auto& i : list)
                logMessage(LV_diagnostic,
                           "'%s' ",
                           i->mName.C_Str());

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
void SLAssimpImporter::loadSkeleton(SLAnimManager& animManager, SLJoint* parent, aiNode* node)
{
    if (!node)
        return;

    SLJoint* joint;
    SLstring name = node->mName.C_Str();

    if (!parent)
    {
        logMessage(LV_normal, "Loading skeleton skeleton.\n");
        _skeleton = new SLAnimSkeleton;
        animManager.skeletons().push_back(_skeleton);
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
        loadSkeleton(animManager, joint, node->mChildren[i]);
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadMaterial loads the AssImp aiMat an returns the SLMaterial.
The materials and textures are added to the SLScene aiMat and texture
vectors.
*/
SLMaterial* SLAssimpImporter::loadMaterial(SLAssetManager* am,
                                           SLint           index,
                                           aiMaterial*     aiMat,
                                           const SLstring& modelPath,
                                           const SLstring& texturePath,
                                           SLSkybox*       skybox,
                                           float           ambientFactor,
                                           SLbool          forceCookTorranceRM,
                                           SLbool          deleteTexImgAfterBuild)
{
    PROFILE_FUNCTION();

    // Get the materials name
    aiString matName;
    aiMat->Get(AI_MATKEY_NAME, matName);
    SLstring name = matName.data;
    if (name.empty()) name = "Import Material";

    // Create SLMaterial instance. It is also added to the SLScene::_materials vector
    SLMaterial* slMat = new SLMaterial(am, name.c_str());

    // load all the textures for this aiMat and add it to the aiMat vector
    for (int tt = aiTextureType_NONE; tt <= aiTextureType_UNKNOWN; ++tt)
    {
        aiTextureType aiTexType = (aiTextureType)tt;

        if (aiMat->GetTextureCount(aiTexType) > 0)
        {
            aiString         aiPath("");
            aiTextureMapping mappingType = aiTextureMapping_UV;
            SLuint           uvIndex     = 0;

            aiMat->GetTexture(aiTexType,
                              0,
                              &aiPath,
                              &mappingType,
                              &uvIndex,
                              nullptr,
                              nullptr,
                              nullptr);

            SLTextureType slTexType = TT_unknown;

            switch (aiTexType)
            {
                case aiTextureType_DIFFUSE: slTexType = TT_diffuse; break;
                case aiTextureType_NORMALS: slTexType = TT_normal; break;
                case aiTextureType_SPECULAR: slTexType = TT_specular; break;
                case aiTextureType_HEIGHT: slTexType = TT_height; break;
                case aiTextureType_OPACITY: slTexType = TT_diffuse; break;
                case aiTextureType_EMISSIVE: slTexType = TT_emissive; break;
                case aiTextureType_LIGHTMAP:
                {
                    // Check if the glTF occlusion texture is within a occlusionRoughnessMetallic texture
                    aiString fileRoughnessMetallic;
                    aiMat->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE,
                                      &fileRoughnessMetallic);
                    SLstring occRghMtlTex = checkFilePath(modelPath,
                                                          texturePath,
                                                          fileRoughnessMetallic.data,
                                                          false);
                    SLstring occlusionTex = checkFilePath(modelPath,
                                                          texturePath,
                                                          aiPath.data,
                                                          false);
                    if (occRghMtlTex == occlusionTex)
                        slTexType = TT_occluRoughMetal;
                    else
                        slTexType = TT_occlusion;

                    // Erleb-AR occulsion map use uvIndex 1
                    string filenameWOExt = Utils::getFileNameWOExt(aiPath.data);
                    if (Utils::startsWithString(filenameWOExt, "AO") ||
                        Utils::endsWithString(filenameWOExt, "AO"))
                        uvIndex = 1;

                    break; // glTF stores AO maps as light maps
                }
                case aiTextureType_AMBIENT_OCCLUSION:
                {
                    // Check if the glTF occlusion texture is within a occlusionRoughnessMetallic texture
                    aiString fileRoughnessMetallic;
                    aiMat->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE,
                                      &fileRoughnessMetallic);
                    SLstring occRghMtlTex = checkFilePath(modelPath,
                                                          texturePath,
                                                          fileRoughnessMetallic.data,
                                                          false);
                    SLstring occlusionTex = checkFilePath(modelPath,
                                                          texturePath,
                                                          aiPath.data,
                                                          false);
                    if (occRghMtlTex == occlusionTex)
                        slTexType = TT_occluRoughMetal;
                    else
                        slTexType = TT_occlusion;
                    break; // glTF stores AO maps as light maps
                }
                case aiTextureType_UNKNOWN:
                {
                    // Check if the unknown texture is a roughnessMetallic texture
                    aiString fileMetallicRoughness;
                    aiMat->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE,
                                      &fileMetallicRoughness);
                    SLstring rghMtlTex  = checkFilePath(modelPath,
                                                       texturePath,
                                                       fileMetallicRoughness.data,
                                                       false);
                    SLstring unknownTex = checkFilePath(modelPath,
                                                        texturePath,
                                                        aiPath.data,
                                                        false);
                    if (rghMtlTex == unknownTex)
                    {
                        // Check if the  roughnessMetallic texture also is the occlusion texture
                        aiString fileOcclusion;
                        aiMat->GetTexture(aiTextureType_LIGHTMAP,
                                          0,
                                          &fileOcclusion);
                        SLstring occlusionTex = checkFilePath(modelPath,
                                                              texturePath,
                                                              fileOcclusion.data,
                                                              false);
                        if (rghMtlTex == occlusionTex)
                            slTexType = TT_unknown; // Don't load twice. The occlusionRoughnessMetallic texture will be loaded as aiTextureType_LIGHTMAP
                        else
                            slTexType = TT_roughMetal;
                    }
                    else
                        slTexType = TT_unknown;
                    break;
                }
                default: break;
            }

            SLstring texFile = checkFilePath(modelPath, texturePath, aiPath.data);

            // Only color texture are loaded so far
            // For normal maps we have to adjust first the normal and tangent generation
            if (slTexType == TT_diffuse ||
                slTexType == TT_normal ||
                slTexType == TT_occlusion ||
                slTexType == TT_emissive ||
                slTexType == TT_roughMetal ||
                slTexType == TT_occluRoughMetal)
            {
                SLGLTexture* slTex = loadTexture(am,
                                                 texFile,
                                                 slTexType,
                                                 uvIndex,
                                                 deleteTexImgAfterBuild);
                slMat->addTexture(slTex);
            }
        }
    }

    // get color data
    aiColor3D ambient, diffuse, specular, emissive;
    SLfloat   shininess, refracti, reflectivity, opacity, roughness = -1, metalness = -1;
    aiMat->Get(AI_MATKEY_COLOR_AMBIENT, ambient);
    aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
    aiMat->Get(AI_MATKEY_COLOR_SPECULAR, specular);
    aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
    aiMat->Get(AI_MATKEY_SHININESS, shininess);
    aiMat->Get(AI_MATKEY_REFRACTI, refracti);
    aiMat->Get(AI_MATKEY_REFLECTIVITY, reflectivity);
    aiMat->Get(AI_MATKEY_OPACITY, opacity);
    aiMat->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR, metalness);
    aiMat->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR, roughness);
    aiString texRoughnessMetallic;

    // increase shininess if specular color is not low.
    // The aiMat will otherwise be too bright
    if (specular.r > 0.5f &&
        specular.g > 0.5f &&
        specular.b > 0.5f &&
        shininess < 0.01f)
        shininess = 10.0f;

    // set color data
    if (ambientFactor > 0.0f)
        slMat->ambient(SLCol4f(diffuse.r * ambientFactor,
                               diffuse.g * ambientFactor,
                               diffuse.b * ambientFactor));
    else
        slMat->ambient(SLCol4f(ambient.r, ambient.g, ambient.b));

    slMat->diffuse(SLCol4f(diffuse.r, diffuse.g, diffuse.b));
    slMat->specular(SLCol4f(specular.r, specular.g, specular.b));
    slMat->emissive(SLCol4f(emissive.r, emissive.g, emissive.b));
    slMat->shininess(shininess);
    slMat->roughness(roughness);
    slMat->metalness(metalness);

    // Switch lighting model to PBR (RM_CookTorrance) only if PBR textures are used.
    // PBR without must be set by additional setter call
    if (slMat->hasTextureType(TT_roughness) ||
        slMat->hasTextureType(TT_metallic) ||
        slMat->hasTextureType(TT_roughMetal) ||
        slMat->hasTextureType(TT_occluRoughMetal) ||
        forceCookTorranceRM)
    {
        slMat->reflectionModel(RM_CookTorrance);
        slMat->skybox(skybox);

        if (roughness == -1.0f)
            slMat->roughness(1.0f);

        if (metalness == -1.0f)
            slMat->metalness(0.0f);
    }
    else
    {
        slMat->reflectionModel(RM_BlinnPhong);
    }

    return slMat;
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadTexture loads the AssImp texture an returns the SLGLTexture
*/
SLGLTexture* SLAssimpImporter::loadTexture(SLAssetManager* assetMgr,
                                           SLstring&       textureFile,
                                           SLTextureType   texType,
                                           SLuint          uvIndex,
                                           SLbool          deleteTexImgAfterBuild)
{
    PROFILE_FUNCTION();

    SLVGLTexture& allLoadedTex = assetMgr->textures();

    // return if a texture with the same file already exists
    for (auto& i : allLoadedTex)
        if (i->url() == textureFile)
            return i;

    SLint minificationFilter = texType == TT_occlusion ? GL_LINEAR : SL_ANISOTROPY_MAX;

    // Create the new texture. It is also push back to SLScene::_textures
    SLGLTexture* texture = new SLGLTexture(assetMgr,
                                           textureFile,
                                           minificationFilter,
                                           GL_LINEAR,
                                           texType);
    texture->uvIndex(uvIndex);

    // if texture images get deleted after build you can't do ray tracing
    if (deleteTexImgAfterBuild)
        texture->deleteImageAfterBuild(true);

    return texture;
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadMesh creates a new SLMesh an copies the meshs vertex data and
triangle face indices. Normals & tangents are not loaded. They are calculated
in SLMesh.
*/
SLMesh* SLAssimpImporter::loadMesh(SLAssetManager* am, aiMesh* mesh)
{
    PROFILE_FUNCTION();

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
        // SL_LOG("SLAssimpImporter::loadMesh:  Mesh contains multiple primitive types: %s, Lines: %d, Points: %d",
        //        mesh->mName.C_Str(),
        //        numLines,
        //        numPoints);

        // Prioritize triangles over lines over points
        if (numTriangles && numLines) numLines = 0;
        if (numTriangles && numPoints) numPoints = 0;
        if (numLines && numPoints) numPoints = 0;
    }

    if (numPolygons > 0)
    {
        SL_LOG("SLAssimpImporter::loadMesh:  Mesh contains polygons: %s",
               mesh->mName.C_Str());
        return nullptr;
    }

    // We only load meshes that contain triangles or lines
    if (mesh->mNumVertices == 0)
    {
        SL_LOG("SLAssimpImporter::loadMesh:  Mesh has no vertices: %s",
               mesh->mName.C_Str());
        return nullptr;
    }

    // We only load meshes that contain triangles or lines
    if (numTriangles == 0 && numLines == 0 && numPoints == 0)
    {
        SL_LOG("SLAssimpImporter::loadMesh:  Mesh has has no triangles nor lines nor points: %s",
               mesh->mName.C_Str());
        return nullptr;
    }

    // create a new mesh.
    // The mesh pointer is added automatically to the SLScene::meshes vector.
    SLstring name = mesh->mName.data;
    SLMesh*  m    = new SLMesh(am, name.empty() ? "Imported Mesh" : name);

    // Set primitive type
    if (numTriangles) m->primitive(SLGLPrimitiveType::PT_triangles);
    if (numLines) m->primitive(SLGLPrimitiveType::PT_lines);
    if (numPoints) m->primitive(SLGLPrimitiveType::PT_points);

    // Create position & normal vector
    m->P.clear();
    m->P.resize(mesh->mNumVertices);

    // Create normal vector for triangle primitive types
    if (mesh->HasNormals() && numTriangles)
    {
        m->N.clear();
        m->N.resize(m->P.size());
    }

    // Allocate 1st tex. coord. vector if needed
    if (mesh->HasTextureCoords(0) && numTriangles)
    {
        m->UV[0].clear();
        m->UV[0].resize(m->P.size());
    }

    // Allocate 2nd texture coordinate vector if needed
    // Some models use multiple textures with different uv's
    if (mesh->HasTextureCoords(1) && numTriangles)
    {
        m->UV[1].clear();
        m->UV[1].resize(m->P.size());
    }

    // copy vertex positions & tex. coord.
    for (SLuint i = 0; i < m->P.size(); ++i)
    {
        m->P[i].set(mesh->mVertices[i].x,
                    mesh->mVertices[i].y,
                    mesh->mVertices[i].z);
        if (!m->N.empty())
            m->N[i].set(mesh->mNormals[i].x,
                        mesh->mNormals[i].y,
                        mesh->mNormals[i].z);
        if (!m->UV[0].empty())
            m->UV[0][i].set(mesh->mTextureCoords[0][i].x,
                            mesh->mTextureCoords[0][i].y);
        if (!m->UV[1].empty())
            m->UV[1][i].set(mesh->mTextureCoords[1][i].x,
                            mesh->mTextureCoords[1][i].y);
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
                for (SLuint nW = 0; nW < joint->mNumWeights; nW++)
                {
                    // add the weight
                    SLuint  vertId = joint->mWeights[nW].mVertexId;
                    SLfloat weight = joint->mWeights[nW].mWeight;

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
                SL_LOG("Failed to load joint of skeleton in SLAssimpImporter::loadMesh: %s",
                       joint->mName.C_Str());
                // return nullptr;
            }
        }
    }

    return m;
}
//-----------------------------------------------------------------------------
/*!
SLAssimpImporter::loadNodesRec loads the scene graph node tree recursively.
*/
SLNode* SLAssimpImporter::loadNodesRec(SLNode*    curNode,    //!< Pointer to the current node. Pass nullptr for root node
                                       aiNode*    node,       //!< The according assimp node. Pass nullptr for root node
                                       SLMeshMap& meshes,     //!< Reference to the meshes vector
                                       SLbool     loadMeshesOnly) //!< Only load nodes with meshes
{
    PROFILE_FUNCTION();

    // we're at the root
    if (!curNode)
        curNode = new SLNode(node->mName.data);

    // load local transform
    aiMatrix4x4* M = &node->mTransformation;

    // clang-format off
    SLMat4f SLM(M->a1, M->a2, M->a3, M->a4,
                M->b1, M->b2, M->b3, M->b4,
                M->c1, M->c2, M->c3, M->c4,
                M->d1, M->d2, M->d3, M->d4);
    // clang-format on

    curNode->om(SLM);

    // New: Add only one mesh per node so that they can be sorted by material
    // If a mesh has multiple meshes add a sub-node for each mesh
    if (node->mNumMeshes > 1)
    {
        for (SLuint i = 0; i < node->mNumMeshes; ++i)
        {
            // Only add meshes that were added to the meshMap (triangle meshes)
            if (meshes.count((SLint)node->mMeshes[i]))
            {
                SLstring nodeMeshName = node->mName.data;
                nodeMeshName += "-";
                nodeMeshName += meshes[(SLint)node->mMeshes[i]]->name();
                SLNode* child = new SLNode(nodeMeshName);
                curNode->addChild(child);
                child->addMesh(meshes[(SLint)node->mMeshes[i]]);
            }
        }
    }
    else if (node->mNumMeshes == 1)
    {
        // Only add meshes that were added to the meshMap (triangle meshes)
        if (meshes.count((SLint)node->mMeshes[0]))
            curNode->addMesh(meshes[(SLint)node->mMeshes[0]]);
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
SLAnimation* SLAssimpImporter::loadAnimation(SLAnimManager& animManager, aiAnimation* anim)
{
    ostringstream oss;
    oss << "unnamed_anim_" << animManager.allAnimNames().size();
    SLstring animName        = oss.str();
    SLfloat  animTicksPerSec = (anim->mTicksPerSecond < 0.0001f)
                                 ? 30.0f
                                 : (SLfloat)anim->mTicksPerSecond;
    SLfloat  animDuration    = (SLfloat)anim->mDuration / animTicksPerSec;

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
        result = _skeleton->createAnimation(animManager, animName, animDuration);
    else
    {
        result = animManager.createNodeAnimation(animName, animDuration);
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
        SLNodeAnimTrack* track = result->createNodeAnimTrack(id);

        // this is a node animation only, so we add a reference to the affected node to the track
        if (affectedNode && !isSkeletonAnim)
        {
            track->animatedNode(affectedNode);
        }

        KeyframeMap keyframes;

        // add position keys
        for (SLuint iK = 0; iK < channel->mNumPositionKeys; iK++)
        {
            SLfloat time    = (SLfloat)channel->mPositionKeys[iK].mTime;
            keyframes[time] = SLImportKeyframe(&channel->mPositionKeys[iK], nullptr, nullptr);
        }

        // add rotation keys
        for (SLuint iK = 0; iK < channel->mNumRotationKeys; iK++)
        {
            SLfloat time = (SLfloat)channel->mRotationKeys[iK].mTime;

            if (keyframes.find(time) == keyframes.end())
                keyframes[time] = SLImportKeyframe(nullptr, &channel->mRotationKeys[iK], nullptr);
            else
            {
                // @todo this shouldn't abort but just throw an exception
                assert(keyframes[time].rotation == nullptr && "There were two rotation keys assigned to the same timestamp.");
                keyframes[time].rotation = &channel->mRotationKeys[iK];
            }
        }

        // add scaling keys
        for (SLuint iK = 0; iK < channel->mNumScalingKeys; iK++)
        {
            SLfloat time = (SLfloat)channel->mScalingKeys[iK].mTime;

            if (keyframes.find(time) == keyframes.end())
                keyframes[time] = SLImportKeyframe(nullptr, nullptr, &channel->mScalingKeys[iK]);
            else
            {
                // @todo this shouldn't abort but just throw an exception
                assert(keyframes[time].scaling == nullptr && "There were two scaling keys assigned to the same timestamp.");
                keyframes[time].scaling = &channel->mScalingKeys[iK];
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
SLstring SLAssimpImporter::checkFilePath(const SLstring& modelPath,
                                         const SLstring& texturePath,
                                         SLstring        aiTexFile,
                                         bool            showWarning)
{
    // Check path & file combination
    SLstring pathFile = modelPath + aiTexFile;
    if (SLFileStorage::exists(pathFile, IOK_generic))
        return pathFile;

    // Check file alone
    if (SLFileStorage::exists(aiTexFile, IOK_generic))
        return aiTexFile;

    // Check path & file combination
    pathFile = modelPath + Utils::getFileName(aiTexFile);
    if (SLFileStorage::exists(pathFile, IOK_generic))
        return pathFile;

    if (showWarning)
    {
        SLstring msg = "SLAssimpImporter: Texture file not found: \n" + aiTexFile +
                       "\non model path: " + modelPath + "\n";
        SL_WARN_MSG(msg.c_str());
    }

    // Return path for texture not found image;
    return texturePath + "TexNotFound.png";
}
//-----------------------------------------------------------------------------

#endif // SL_BUILD_WITH_ASSIMP
