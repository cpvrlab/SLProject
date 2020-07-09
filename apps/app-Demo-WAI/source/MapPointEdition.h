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
public:
    MapEdition(SLSceneView* sv, SLNode* mappointNode, WAIMap * map, SLstring shaderDir);
    ~MapEdition() override;

    void updateKFVidMatching(std::vector<int>* kFVidMatching);
    void selectAllMap();
    void selectNMatched(std::vector<bool> nmatches);
    void selectByVid(std::vector<bool> vid);
    void filterVisibleKeyframes(std::vector<WAIKeyFrame*>& kfs, std::vector<WAIMapPoint*> activeMps);

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

    void updateMeshes(std::string                      name,
                      const std::vector<WAIMapPoint*>& pts,
                      const std::vector<WAIKeyFrame*>& kfs,
                      SLPoints*&                       mesh,
                      SLMaterial*&                     material);

private:
    SLSceneView*         _sv;
    SLCamera*            _camera;
    SLGLProgram*         _prog = nullptr;
    SLMaterial*          _green;

    SLNode*              _mapNode;
    SLNode*              _workingNode;
    SLNode*              _kfNode;
    SLNode*              _mpNode;

    SLPoints*            _mesh = nullptr;
    std::vector<int>     _meshToMP;

    WAIMap*              _map;
    vector<WAIMapPoint*> _mappoints;
    vector<WAIKeyFrame*> _keyframes;
    vector<WAIMapPoint*> _activeMapPoints;
    vector<WAIMapPoint*> _temporaryMapPoints;
    vector<WAIKeyFrame*> _activeKeyframes;

    bool                 _keyframeMode  = false;
    bool                 _transformMode = false;

    std::vector<std::vector<WAIKeyFrame*>> _vidToKeyframes;
    std::map<WAIKeyFrame*, int>            _keyFramesToVid;
};

#endif
