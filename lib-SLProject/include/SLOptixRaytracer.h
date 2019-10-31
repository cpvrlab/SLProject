//#############################################################################
//  File:      SLOptixRaytracer.h
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLOPTIXRAYTRACER_H
#define SLPROJECT_SLOPTIXRAYTRACER_H
#include <SLEventHandler.h>
#include <SLGLTexture.h>
#include <optix_types.h>
#include <cuda.h>
#include <SLOptixDefinitions.h>
#include "SLCudaBuffer.h"

class SLScene;
class SLSceneView;
class SLRay;
class SLMaterial;
class SLCamera;

//-----------------------------------------------------------------------------
//! SLOptixRaytracer hold all the methods for Whitted style Ray Tracing.
class SLOptixRaytracer : public SLRaytracer
{
public:
    SLOptixRaytracer();
    ~SLOptixRaytracer() override;

    // setup raytracer
    void setupOptix();
    void setupScene();
    void updateScene(SLSceneView* sv);

    // ray tracer functions
    SLbool  renderClassic();
    void renderImage() override;

protected:
    void _createContext();
    OptixModule _createModule(std::string);
    OptixProgramGroup _createProgram(OptixProgramGroupDesc);
    OptixPipeline _createPipeline(OptixProgramGroup *, unsigned int);
    OptixShaderBindingTable _createShaderBindingTable();
    OptixTraversableHandle _createMeshAccelerationStructure(SLMesh);

    OptixDeviceContext          _context{};
    CUstream                    _stream{};

    SLCudaBuffer<uchar4>            _imageBuffer = SLCudaBuffer<uchar4>();
    SLCudaBuffer<Params>            _paramsBuffer = SLCudaBuffer<Params>();
    SLCudaBuffer<RayGenSbtRecord>   _rayGenBuffer = SLCudaBuffer<RayGenSbtRecord>();
    SLCudaBuffer<MissSbtRecord>     _missBuffer = SLCudaBuffer<MissSbtRecord>();
    SLCudaBuffer<HitSbtRecord>      _hitBuffer = SLCudaBuffer<HitSbtRecord>();

    OptixModule                 _cameraModule{};
    OptixModule                 _shadingModule{};
    OptixModuleCompileOptions   _module_compile_options{};
    OptixPipelineCompileOptions _pipeline_compile_options{};
    OptixPipeline               _pipeline{};

    OptixProgramGroup           _raygen_prog_group{};
    OptixProgramGroup           _radiance_miss_group{};
    OptixProgramGroup           _occlusion_miss_group{};
    OptixProgramGroup           _radiance_hit_group{};
    OptixProgramGroup           _occlusion_hit_group{};

    OptixShaderBindingTable     _sbt{};
    OptixTraversableHandle      _handle{};
    Params                      _params{};
    Params*                     _d_params{};
    Light*                      _lights{};
    Light*                      _d_lights{};
};
#endif //SLPROJECT_SLOPTIXRAYTRACER_H
