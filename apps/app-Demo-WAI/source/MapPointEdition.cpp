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

    _activeSet = _mappoints;
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
    for (int i = 0; i < kFVidMatching->size(); i++)
    {
        if (kFVidMatching->at(i) == -1)
            continue;
        if (kFVidMatching->at(i) >= _vidToKeyframes.size())
        {
            _vidToKeyframes.resize(kFVidMatching->at(i) + 1);
        }

        int j;
        for (j = 0; j < _keyframes.size() && _keyframes[j]->mnId != i; j++);

        if (j == _keyframes.size())
        {
            std::cout << "kf with mnid " << i << "not found" << std::endl;
            continue;
        }
        _vidToKeyframes[kFVidMatching->at(i)].push_back(_keyframes[j]); //add keyframe i to kfSet[videoId]
    }
}

void MapEdition::selectAllMap()
{
    _activeSet = _mappoints;
    updateMapPointsMeshes("current map points", _activeSet, _mesh, _green);
}

void MapEdition::selectNMatched(int n)
{
    std::map<WAIMapPoint*, int> counters;

    for (int vid = 0; vid < _vidToKeyframes.size(); vid++)
    {
        for (int i = 0; i < _vidToKeyframes[vid].size(); i++)
        {
            WAIKeyFrame* kf = _vidToKeyframes[vid][i];

            std::vector<WAIMapPoint*> mps = kf->GetMapPointMatches();
            for (WAIMapPoint* mp : mps)
            {
                if (mp == nullptr || mp->isBad())
                    continue;

                auto it = counters.find(mp);
                if (it != counters.end())
                    counters.insert(std::pair<WAIMapPoint*, int>(mp, it->second + 1));
                else
                    counters.insert(std::pair<WAIMapPoint*, int>(mp, 1));
            }
        }
    }

    std::vector<WAIMapPoint*> mpvec;
    for (auto it = counters.begin(); it != counters.end(); it++)
    {
        if (it->second == n)
            mpvec.push_back(it->first);
    }

    _activeSet = mpvec;
    updateMapPointsMeshes("current map points", _activeSet, _mesh, _green);
}

void MapEdition::selectByVid(int id)
{
    std::set<WAIMapPoint*> mpset;
    int                    lastIdx = 0;
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

    std::vector<WAIMapPoint*> mpvec;
    mpvec.resize(mpset.size());
    std::copy(mpset.begin(), mpset.end(), mpvec.begin());
    _activeSet = mpvec;
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

        mesh = new SLPoints(nullptr, points, normals, name, material);
        _workingNode->addMesh(mesh);
        updateAABBRec();
        _sv->s()->selectNodeMesh(_workingNode, mesh);
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
