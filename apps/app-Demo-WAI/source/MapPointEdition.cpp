#include <MapPointEdition.h>
#include <SLKeyframeCamera.h>
#include <SLAssetManager.h>
#include <SLMaterial.h>
#include <SLPoints.h>
#include <WAISlam.h>

MapEdition::MapEdition(SLSceneView* sv, SLNode* mappointNode, WAIMap* map, SLstring shaderDir)
  : SLTransformNode(sv, mappointNode, shaderDir)
{
    _sv        = sv;
    _mapNode   = mappointNode;
    _map       = map;
    _mappoints = _map->GetAllMapPoints();
    _keyframes = _map->GetAllKeyFrames();

    _temporaryMapPoints = _mappoints;
    _activeMapPoints    = _mappoints;
    _activeKeyframes    = _keyframes;

    _kfNode      = new SLNode("kf working node");
    _mpNode      = new SLNode("map working node");
    _workingNode = new SLNode("editor node");
    _workingNode->addChild(_kfNode);
    _workingNode->addChild(_mpNode);
    _mapNode->addChild(_workingNode);

    setKeyframeMode(false);

    _prog = new SLGLGenericProgram(nullptr, shaderDir + "ColorUniformPoint.vert", shaderDir + "Color.frag");
    _prog->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));

    _green = new SLMaterial(nullptr, "Green Opaque", SLCol4f::GREEN, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);

    updateMeshes("current map points", _activeMapPoints, _activeKeyframes, _mesh, _green);
}

MapEdition::~MapEdition()
{
    deleteMesh(_mesh);
    _kfNode->deleteChildren();
    _mapNode->deleteChild(_workingNode);
    delete _prog;
    delete _green;
}

void MapEdition::updateKFVidMatching(std::vector<int>* kFVidMatching)
{
    _vidToKeyframes.clear();
    _keyFramesToVid.clear();

    for (int i = 0; i < kFVidMatching->size(); i++)
    {
        if (kFVidMatching->at(i) == -1)
            continue;

        if (kFVidMatching->at(i) >= _vidToKeyframes.size())
        {
            _vidToKeyframes.resize(kFVidMatching->at(i) + 1);
        }

        int j;
        for (j = 0; j < _keyframes.size() && _keyframes[j]->mnId != i; j++)
            ;

        if (j == _keyframes.size())
        {
            std::cout << "kf with mnid " << i << "not found" << std::endl;
            continue;
        }
        _vidToKeyframes[kFVidMatching->at(i)].push_back(_keyframes[j]); //add keyframe i to kfSet[videoId]
        _keyFramesToVid.insert(std::pair<WAIKeyFrame*, int>(_keyframes[j], kFVidMatching->at(i)));
    }
}

void MapEdition::selectAllMap()
{
    _activeMapPoints    = _mappoints;
    _temporaryMapPoints = _mappoints;
    _activeKeyframes    = _keyframes;
    updateMeshes("current map points", _activeMapPoints, _activeKeyframes, _mesh, _green);
}

void MapEdition::selectNMatched(std::vector<bool> nmatches)
{
    std::vector<std::set<int>> counters;
    counters.resize(_temporaryMapPoints.size());

    for (int i = 0; i < _temporaryMapPoints.size(); i++)
    {
        std::map<WAIKeyFrame*, size_t> obs = _temporaryMapPoints[i]->GetObservations();
        for (auto it = obs.begin(); it != obs.end(); it++)
        {
            WAIKeyFrame* kf = it->first;

            auto kfvid = _keyFramesToVid.find(kf);
            if (kfvid != _keyFramesToVid.end())
            {
                counters[i].insert(kfvid->second);
            }
        }
    }

    std::vector<WAIMapPoint*> mpvec;
    for (int i = 0; i < counters.size(); i++)
    {
        for (int j = 0; j < nmatches.size(); j++)
        {
            if (nmatches[j] && counters[i].size() == j)
            {
                mpvec.push_back(_temporaryMapPoints[i]);
                break;
            }
        }
    }

    _activeMapPoints = mpvec;
    filterVisibleKeyframes(_activeKeyframes, _activeMapPoints);
    updateMeshes("current map points", _activeMapPoints, _activeKeyframes, _mesh, _green);
}

void MapEdition::selectByVid(std::vector<bool> vid)
{
    std::set<WAIMapPoint*> mpset;
    int                    lastIdx = 0;
    for (int id = 0; id < vid.size(); id++)
    {
        if (vid[id])
        {
            for (int i = 0; i < _vidToKeyframes[id].size(); i++)
            {
                WAIKeyFrame* kf = _vidToKeyframes[id][i];
                if (kf->isBad())
                    continue;

                std::vector<WAIMapPoint*> mps = kf->GetMapPointMatches();
                for (WAIMapPoint* mp : mps)
                {
                    if (mp == nullptr || mp->isBad())
                        continue;
                    mpset.insert(mp);
                }
            }
        }
    }

    std::vector<WAIMapPoint*> mpvec;
    mpvec.resize(mpset.size());
    std::copy(mpset.begin(), mpset.end(), mpvec.begin());
    _temporaryMapPoints = mpvec;
    _activeMapPoints    = _temporaryMapPoints;

    filterVisibleKeyframes(_activeKeyframes, _activeMapPoints);
    updateMeshes("current map points", _activeMapPoints, _activeKeyframes, _mesh, _green);
}

void MapEdition::updateVisualization()
{
    updateMeshes("current map points", _activeMapPoints, _activeKeyframes, _mesh, _green);
}

void MapEdition::setKeyframeMode(bool v)
{
    if (v)
    {
        _keyframeMode = true;
        _mpNode->setDrawBitsRec(SL_DB_NOTSELECTABLE, true);
        _mpNode->setDrawBitsRec(SL_DB_HIDDEN, true);
        _kfNode->setDrawBitsRec(SL_DB_NOTSELECTABLE, false);
    }
    else
    {
        _keyframeMode = false;
        _mpNode->setDrawBitsRec(SL_DB_NOTSELECTABLE, false);
        _mpNode->setDrawBitsRec(SL_DB_HIDDEN, false);
        _kfNode->setDrawBitsRec(SL_DB_NOTSELECTABLE, true);
    }
}

void MapEdition::updateMeshes(std::string                      name,
                              const std::vector<WAIMapPoint*>& pts,
                              const std::vector<WAIKeyFrame*>& kfs,
                              SLPoints*&                       mesh,
                              SLMaterial*&                     material)
{
    //remove old mesh, if it exists
    deleteMesh(mesh);
    unsigned int i = 0;
    _meshToMP.clear();

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        std::vector<SLVec3f> points, normals;
        for (auto mapPt : pts)
        {
            i++;
            if (!mapPt || mapPt->isBad())
                continue;

            _meshToMP.push_back(i - 1);

            WAI::V3 wP = mapPt->worldPosVec();
            WAI::V3 wN = mapPt->normalVec();
            points.push_back(SLVec3f(wP.x, wP.y, wP.z));
            normals.push_back(SLVec3f(wN.x, wN.y, wN.z));
        }

        if (points.size() > 0)
        {
            mesh = new SLPoints(nullptr, points, normals, name, material);
            _mpNode->addMesh(mesh);
            _mpNode->updateAABBRec();
        }
    }

    _kfNode->deleteChildren();

    i = 0;
    for (WAIKeyFrame* kf : kfs)
    {
        i++;

        if (kf->isBad())
            continue;

        SLKeyframeCamera* cam = new SLKeyframeCamera(std::to_string(i - 1));

        cv::Mat Twc = kf->getObjectMatrix();

        SLMat4f om;
        om.setMatrix(Twc.at<float>(0, 0),
                     -Twc.at<float>(0, 1),
                     -Twc.at<float>(0, 2),
                     Twc.at<float>(0, 3),
                     Twc.at<float>(1, 0),
                     -Twc.at<float>(1, 1),
                     -Twc.at<float>(1, 2),
                     Twc.at<float>(1, 3),
                     Twc.at<float>(2, 0),
                     -Twc.at<float>(2, 1),
                     -Twc.at<float>(2, 2),
                     Twc.at<float>(2, 3),
                     Twc.at<float>(3, 0),
                     -Twc.at<float>(3, 1),
                     -Twc.at<float>(3, 2),
                     Twc.at<float>(3, 3));

        cam->om(om);

        //calculate vertical field of view
        SLfloat fy     = (SLfloat)kf->fy;
        SLfloat cy     = (SLfloat)kf->cy;
        SLfloat fovDeg = 2 * (SLfloat)atan2(cy, fy) * Utils::RAD2DEG;
        cam->fov(fovDeg);
        cam->focalDist(0.11f);
        cam->clipNear(0.1f);
        cam->clipFar(1000.0f);

        _kfNode->addChild(cam);
    }

    _kfNode->updateAABBRec();
}

void MapEdition::editMode(SLNodeEditMode editMode)
{
    _transformMode = true;
    SLTransformNode::editMode(editMode);
}

void MapEdition::deleteMesh(SLPoints*& mesh)
{
    if (mesh)
    {
        if (_mpNode->removeMesh(mesh))
        {
            delete mesh;
            mesh = nullptr;
            _sv->s()->selectNodeMesh(nullptr, nullptr);
        }
    }
}

SLbool MapEdition::onKeyPress(const SLKey key, const SLKey mod)
{
    (void)key;
    (void)mod;

    return false;
}

void MapEdition::filterVisibleKeyframes(std::vector<WAIKeyFrame*>& kfs, std::vector<WAIMapPoint*> activeMps)
{
    kfs.clear();
    std::map<WAIKeyFrame*, int> kfsel;

    for (WAIMapPoint* mp : activeMps)
    {
        if (!mp || mp->isBad())
            continue;

        std::map<WAIKeyFrame*, size_t> obs = mp->GetObservations();
        for (auto it = obs.begin(); it != obs.end(); it++)
        {
            if (it->first->isBad())
                continue;

            auto mit = kfsel.find(it->first);
            if (mit != kfsel.end())
                mit->second += 1;
            else
                kfsel.insert(std::make_pair(it->first, 1));
        }
    }

    for (auto it = kfsel.begin(); it != kfsel.end(); ++it)
    {
        if (it->second > 8)
            kfs.push_back(it->first);
    }
}

SLbool MapEdition::onKeyRelease(const SLKey key, const SLKey mod)
{
    (void)mod;

    if (key == K_delete)
    {
        if (_keyframeMode)
        {
            for (auto it = _sv->s()->selectedNodes().begin(); it != _sv->s()->selectedNodes().end(); it++)
            {
                SLNode* node = *it;
                if (node->parent() == _kfNode)
                {
                    int idx = std::stoi(node->name());
                    if (_activeKeyframes[idx]->mnId == 0)
                    {
                        Utils::log("WARNING", "Try to delete initial keyframe\n");
                        continue;
                    }
                    _activeKeyframes[idx]->SetBadFlag();
                    _map->EraseKeyFrame(_activeKeyframes[idx]);
                    _map->GetKeyFrameDB()->erase(_activeKeyframes[idx]);
                }
            }
        }
        else
        {
            for (unsigned int i = 0; i < _mesh->IS32.size(); i++)
            {
                _activeMapPoints[_meshToMP[_mesh->IS32[i]]]->SetBadFlag();
            }

            for (unsigned int i = 0; i < _keyframes.size(); i++)
            {
                if (_keyframes[i]->isBad()) continue;

                unsigned int              nbGood = 0;
                std::vector<WAIMapPoint*> mps    = _keyframes[i]->GetMapPointMatches();

                for (unsigned int j = 0; j < mps.size(); j++)
                {
                    if (mps[j] && !mps[j]->isBad())
                        nbGood++;
                }
                if (nbGood <= 8 && _keyframes[i]->mnId != 0)
                {
                    _keyframes[i]->SetBadFlag();
                    _map->EraseKeyFrame(_keyframes[i]);
                    _map->GetKeyFrameDB()->erase(_keyframes[i]);
                }
            }
        }
        updateMeshes("current map points", _activeMapPoints, _activeKeyframes, _mesh, _green);
    }
    return false;
}

SLbool MapEdition::onMouseDown(SLMouseButton button,
                               SLint         x,
                               SLint         y,
                               SLKey         mod)
{
    if (_transformMode)
        return SLTransformNode::onMouseDown(button, x, y, mod);
    return false;
}

SLbool MapEdition::onMouseMove(const SLMouseButton button,
                               SLint               x,
                               SLint               y,
                               const SLKey         mod)
{
    if (_transformMode)
        return SLTransformNode::onMouseMove(button, x, y, mod);
    return false;
}

SLbool MapEdition::onMouseUp(SLMouseButton button,
                             SLint         x,
                             SLint         y,
                             SLKey         mod)
{
    if (_transformMode)
        return SLTransformNode::onMouseUp(button, x, y, mod);
    return false;
}
