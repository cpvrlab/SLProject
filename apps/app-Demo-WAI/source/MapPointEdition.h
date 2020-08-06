#ifndef SL_MAP_EDITION_NODE_H
#define SL_MAP_EDITION_NODE_H

#include <SLSceneView.h>
#include <SLTransformNode.h>
#include <WAIMapPoint.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <SLMesh.h>

class MapEdition : public SLTransformNode
{
    struct MapPointsAndMat
    {
        std::vector<WAIMapPoint*> pts;
        SLMaterial*               mat;
    };

public:
    MapEdition(SLSceneView* sv, SLNode* mappointNode, WAIMap* map, SLstring shaderDir);
    ~MapEdition() override;

    void updateKFVidMatching(std::vector<int>* kFVidMatching);
    void selectAllMap();
    void selectNMatched(std::vector<bool> nmatches);
    void selectByVid(std::vector<bool> vid);
    void filterVisibleKeyframes(std::vector<WAIKeyFrame*>& kfs, std::vector<std::vector<WAIMapPoint*>> activeMapPointSets);

    SLbool onKeyPress(const SLKey key, const SLKey mod) override;

    SLbool onKeyRelease(const SLKey key, const SLKey mod) override;

    SLbool onMouseDown(SLMouseButton button,
                       SLint         x,
                       SLint         y,
                       SLKey         mod) override;

    SLbool onMouseUp(SLMouseButton button,
                     SLint         x,
                     SLint         y,
                     SLKey         mod) override;

    SLbool onMouseMove(SLMouseButton button,
                       SLint         x,
                       SLint         y,
                       SLKey         mod) override;

    void editMode(SLNodeEditMode editMode) override;

    void deleteMesh(SLPoints*& mesh);

    void updateVisualization();
    void setKeyframeMode(bool v);

    void updateMeshes(std::string                         name,
                      const vector<vector<WAIMapPoint*>>& ptSets,
                      const std::vector<WAIKeyFrame*>&    kfs,
                      vector<SLPoints*>&                  meshes,
                      vector<SLMaterial*>&                materials);

private:
    SLSceneView* _sv;
    SLCamera*    _camera;
    SLGLProgram* _prog = nullptr;

    SLNode* _mapNode;
    SLNode* _workingNode;
    SLNode* _kfNode;
    SLNode* _mpNode;

    vector<SLPoints*>        _meshes;
    vector<std::vector<int>> _meshesToMP;
    vector<SLMaterial*>      _materials;

    WAIMap*                      _map;
    vector<WAIMapPoint*>         _mappoints;
    vector<WAIKeyFrame*>         _keyframes;
    vector<vector<WAIMapPoint*>> _activeMapPointSets;
    vector<vector<WAIMapPoint*>> _temporaryMapPointSets;
    vector<WAIKeyFrame*>         _activeKeyframes;

    bool _keyframeMode  = false;
    bool _transformMode = false;

    std::vector<std::vector<WAIKeyFrame*>> _vidToKeyframes;
    std::map<WAIKeyFrame*, int>            _keyFramesToVid;
};

#endif
