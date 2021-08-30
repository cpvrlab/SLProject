//#############################################################################
//  File:      SLOptixRaytracer.h
//  Authors:   Nic Dorner
//  Date:      October 2019
//  Authors:   Nic Dorner
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    ifndef SLOPTIXRAYTRACER_H
#        define SLOPTIXRAYTRACER_H
#        include <optix_types.h>
#        include <cuda.h>
#        include <SLOptixDefinitions.h>
#        include <SLRaytracer.h>
#        include <SLOptixCudaBuffer.h>
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

    void destroy();

    // setup raytracer
    virtual void setupOptix();
    virtual void setupScene(SLSceneView* sv, SLAssetManager* am);
    virtual void updateScene(SLSceneView* sv);

    void prepareImage() override;

    // ray tracer functions
    SLbool       renderClassic();
    SLbool       renderDistrib();
    virtual void renderImage(bool updateTextureGL) override;
    void         saveImage() override;

protected:
    void initCompileOptions();

    OptixModule             createModule(string);
    OptixProgramGroup       createProgram(OptixProgramGroupDesc);
    OptixPipeline           createPipeline(OptixProgramGroup*, unsigned int);
    OptixShaderBindingTable createShaderBindingTable(const SLVMesh&, const bool);

    SLOptixCudaBuffer<float4>    _imageBuffer  = SLOptixCudaBuffer<float4>();
    SLOptixCudaBuffer<ortParams> _paramsBuffer = SLOptixCudaBuffer<ortParams>();
    SLOptixCudaBuffer<ortLight>  _lightBuffer  = SLOptixCudaBuffer<ortLight>();

    OptixModule                 _cameraModule{};
    OptixModule                 _shadingModule{};
    OptixModuleCompileOptions   _module_compile_options{};
    OptixPipelineCompileOptions _pipeline_compile_options{};

    OptixTraversableHandle _handle{};
    ortParams              _params{};

    SLOptixCudaBuffer<MissSbtRecord>              _missBuffer              = SLOptixCudaBuffer<MissSbtRecord>();
    SLOptixCudaBuffer<HitSbtRecord>               _hitBuffer               = SLOptixCudaBuffer<HitSbtRecord>();
    SLOptixCudaBuffer<RayGenClassicSbtRecord>     _rayGenClassicBuffer     = SLOptixCudaBuffer<RayGenClassicSbtRecord>();
    SLOptixCudaBuffer<RayGenDistributedSbtRecord> _rayGenDistributedBuffer = SLOptixCudaBuffer<RayGenDistributedSbtRecord>();

    OptixPipeline _pipeline{};

    OptixProgramGroup _pinhole_raygen_prog_group{};
    OptixProgramGroup _lens_raygen_prog_group{};
    OptixProgramGroup _orthographic_raygen_prog_group{};
    OptixProgramGroup _radiance_miss_group{};
    OptixProgramGroup _occlusion_miss_group{};
    OptixProgramGroup _radiance_hit_group{};
    OptixProgramGroup _occlusion_hit_group{};

    OptixShaderBindingTable _sbtClassic{};
    OptixShaderBindingTable _sbtDistributed{};
};
//-----------------------------------------------------------------------------
#    endif // SLOPTIXRAYTRACER_H
#endif     // SL_HAS_OPTIX
