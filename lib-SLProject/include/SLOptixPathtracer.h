//
// Created by nic on 19.12.19.
//

#ifndef SLPROJECT_SLOPTIXPATHTRACER_H
#define SLPROJECT_SLOPTIXPATHTRACER_H

#include <SLOptixRaytracer.h>
#include <curand_kernel.h>

class SLScene;
class SLSceneView;
class SLRay;
class SLMaterial;
class SLCamera;

class SLOptixPathtracer : public SLOptixRaytracer
{
public:
    SLOptixPathtracer();
    ~SLOptixPathtracer();

    // setup path tracer
    void setupOptix() override;
    void setupScene(SLSceneView* sv) override;
    void updateScene(SLSceneView* sv) override;

    // path tracer functions
    SLbool  render();
    void    renderImage() override;

    SLbool getDenoiserEnabled() const {return _denoiserEnabled;}
    SLint samples() const { return _samples;}

    void setDenoiserEnabled(SLbool denoiserEnabled) {_denoiserEnabled = denoiserEnabled;}
    void samples(SLint samples){ _samples = samples; }

protected:
    SLint                     _samples;
    SLCudaBuffer<curandState> _curandBuffer = SLCudaBuffer<curandState>();

private:
    OptixDenoiser _optixDenoiser;
    OptixDenoiserSizes _denoiserSizes;
    SLCudaBuffer<void> _denoserState;
    SLCudaBuffer<void> _scratch;

    OptixShaderBindingTable     _createShaderBindingTable(const SLVMesh&);

    OptixPipeline _path_tracer_pipeline;
    OptixProgramGroup _sample_raygen_prog_group;
    OptixProgramGroup _sample_miss_group;
    OptixProgramGroup _sample_hit_group;
    SLCudaBuffer<RayGenPathtracerSbtRecord>  _rayGenPathtracerBuffer;
    OptixShaderBindingTable _sbtPathtracer;

    //Settings
    SLbool _denoiserEnabled = true;
};

#endif //SLPROJECT_SLOPTIXPATHTRACER_H
