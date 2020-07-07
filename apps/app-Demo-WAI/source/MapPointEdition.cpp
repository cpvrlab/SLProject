#include <MapPointEdition.h>
#include <SLAssetManager.h>
#include <SLMaterial.h>
#include <SLPoints.h>
#include <WAISlam.h>

MapEdition::MapEdition(SLSceneView* sv, SLNode* mappointNode, vector<WAIMapPoint*> mp, vector<WAIKeyFrame*> kf, SLstring shaderDir)
  : SLTransformNode(sv, mappointNode, shaderDir)
{
    _sv        = sv;
    _mapNode   = mappointNode;
    _mappoints = mp;
    _keyframes = kf;

    _workingNode = new SLNode("map editing node");
    _mapNode->addChild(_workingNode);

    _prog = new SLGLGenericProgram(nullptr, shaderDir + "ColorUniformPoint.vert", shaderDir + "Color.frag");
    _prog->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));

    _yellow = new SLMaterial(nullptr, "Yellow Opaque", SLCol4f::YELLOW, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);
    _green  = new SLMaterial(nullptr, "Green Opaque", SLCol4f::GREEN, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);

    _activeSet    = _mappoints;
    _temporarySet = _mappoints;
    updateMapPointsMeshes("current map points", _activeSet, _mesh, _green);
}

MapEdition::~MapEdition()
{
    _mapNode->deleteChild(_workingNode);
    deleteMesh(_mesh);
    delete _prog;
    delete _yellow;
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
    _activeSet    = _mappoints;
    _temporarySet = _mappoints;
    updateMapPointsMeshes("current map points", _activeSet, _mesh, _green);
}

void MapEdition::selectNMatched(std::vector<bool> nmatches)
{
    std::vector<std::set<int>> counters;
    counters.resize(_temporarySet.size());

    for (int i = 0; i < _temporarySet.size(); i++)
    {
        std::map<WAIKeyFrame*, size_t> obs = _temporarySet[i]->GetObservations();
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
                mpvec.push_back(_temporarySet[i]);
                break;
            }
        }
    }

    _activeSet = mpvec;
    updateMapPointsMeshes("current map points", _activeSet, _mesh, _green);
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
    _temporarySet = mpvec;
    _activeSet    = _temporarySet;
    updateMapPointsMeshes("current map points", _activeSet, _mesh, _green);
}

void MapEdition::updateMapPointsMeshes(std::string                      name,
                                       const std::vector<WAIMapPoint*>& pts,
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
            _workingNode->addMesh(mesh);
            updateAABBRec();
            _sv->s()->selectNodeMesh(_workingNode, mesh);
        }
    }
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
        if (_workingNode->removeMesh(mesh))
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

SLbool MapEdition::onKeyRelease(const SLKey key, const SLKey mod)
{
    (void)mod;

    if (key == K_delete)
    {
        for (unsigned int i = 0; i < _mesh->IS32.size(); i++)
        {
            _activeSet[_meshToMP[_mesh->IS32[i]]]->SetBadFlag();
        }
        /*
        for (unsigned int i = 0; i < _keyframes.size(); i++)
        {
            bool deleteKF = true; 
            std::vector<WAIMapPoint*> mps = _keyframes[i]->GetMapPointMatches();
            for (unsigned int j = 0; j < mps.size(); j++)
            {
                if (!mps[j]->isBad())
                    deleteKF = false;
            }
            if (deleteKF)
                _keyframes[i]->SetBadFlag();
        }
        */
        updateMapPointsMeshes("current map points", _activeSet, _mesh, _green);
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
