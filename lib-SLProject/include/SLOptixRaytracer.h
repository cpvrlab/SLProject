//#############################################################################
//  File:      SLOptixRaytracer.h
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    ifndef SLOPTIXRAYTRACER_H
#        define SLOPTIXRAYTRACER_H
#        include <optix_types.h>
#        include <cuda.h>
#        include <SLOptixDefinitions.h>
#        include <SLRaytracer.h>
#        include <SLCudaBuffer.h>
#        include <SLMesh.h>

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
    virtual void setupOptix();
    virtual void setupScene(SLSceneView* sv);
    virtual void updateScene(SLSceneView* sv);

    void prepareImage() override;

    // ray tracer functions
    SLbool       renderClassic();
    SLbool       renderDistrib();
    virtual void renderImage() override;

    void saveImage() override;

    //void drawRay(unsigned int, unsigned int);
    //void removeRays();

    protected:
    void initCompileOptions();

    OptixModule             _createModule(string);
    OptixProgramGroup       _createProgram(OptixProgramGroupDesc);
    OptixPipeline           _createPipeline(OptixProgramGroup*, unsigned int);
    OptixShaderBindingTable _createShaderBindingTable(const SLVMesh&, const bool);

    SLCudaBuffer<float4> _imageBuffer  = SLCudaBuffer<float4>();
    SLCudaBuffer<ortRay>    _lineBuffer   = SLCudaBuffer<ortRay>();
    SLCudaBuffer<ortParams> _paramsBuffer = SLCudaBuffer<ortParams>();
    SLCudaBuffer<ortLight>  _lightBuffer  = SLCudaBuffer<ortLight>();

    OptixModule                 _cameraModule{};
    OptixModule                 _shadingModule{};
    OptixModule                 _traceModule{};
    OptixModuleCompileOptions   _module_compile_options{};
    OptixPipelineCompileOptions _pipeline_compile_options{};

    OptixTraversableHandle _handle{};
    ortParams                 _params{};

    SLCudaBuffer<MissSbtRecord>              _missBuffer              = SLCudaBuffer<MissSbtRecord>();
    SLCudaBuffer<HitSbtRecord>               _hitBuffer               = SLCudaBuffer<HitSbtRecord>();
    SLCudaBuffer<RayGenClassicSbtRecord>     _rayGenClassicBuffer     = SLCudaBuffer<RayGenClassicSbtRecord>();
    SLCudaBuffer<RayGenDistributedSbtRecord> _rayGenDistributedBuffer = SLCudaBuffer<RayGenDistributedSbtRecord>();

    OptixPipeline _pipeline{};

    OptixProgramGroup _pinhole_raygen_prog_group{};
    OptixProgramGroup _lens_raygen_prog_group{};
    OptixProgramGroup _orthographic_raygen_prog_group{};
    OptixProgramGroup _radiance_miss_group{};
    OptixProgramGroup _occlusion_miss_group{};
    OptixProgramGroup _radiance_hit_group{};
    OptixProgramGroup _radiance_line_hit_group;
    OptixProgramGroup _occlusion_hit_group{};
    OptixProgramGroup _occlusion_line_hit_group;

    OptixShaderBindingTable _sbtClassic{};
    OptixShaderBindingTable _sbtDistributed{};
};
//-----------------------------------------------------------------------------
#    endif // SLOPTIXRAYTRACER_H
#endif     // SL_HAS_OPTIX
