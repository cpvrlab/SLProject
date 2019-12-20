//
// Created by nic on 19.12.19.
//

#ifndef SLPROJECT_SLOPTIXPATHTRACER_H
#define SLPROJECT_SLOPTIXPATHTRACER_H

#include <SLOptixRaytracer.h>

class SLScene;
class SLSceneView;
class SLRay;
class SLMaterial;
class SLCamera;

class SLOptixPathtracer : public SLOptixRaytracer
{
public:
    SLOptixPathtracer();
    ~SLOptixPathtracer() override;

    // setup path tracer
    void setupOptix() override;
    void setupScene(SLSceneView* sv) override;
    void updateScene(SLSceneView* sv) override;

    // path tracer functions
    SLbool  render();
    void    renderImage() override;
};

#endif //SLPROJECT_SLOPTIXPATHTRACER_H
