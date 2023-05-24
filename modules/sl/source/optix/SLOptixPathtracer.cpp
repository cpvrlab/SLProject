//#############################################################################
//  File:      SLOptixPathtracer.cpp
//  Authors:   Nic Dorner
//  Date:      Dezember 2019
//  Authors:   Nic Dorner
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    include <SLAssetManager.h>
#    include <SLScene.h>
#    include <SLSceneView.h>
#    include <SLOptix.h>
#    include <SLOptixRaytracer.h>
#    include <SLOptixPathtracer.h>
#    include <GlobalTimer.h>

//-----------------------------------------------------------------------------
SLOptixPathtracer::SLOptixPathtracer()
{
    name("OptiX path tracer");
}
//-----------------------------------------------------------------------------
SLOptixPathtracer::~SLOptixPathtracer()
{
    SL_LOG("Destructor      : ~SLOptixPathtracer");

    try
    {
        OPTIX_CHECK(optixDenoiserDestroy(_optixDenoiser));
    }
    catch (exception e)
    {
        Utils::log("SLProject",
                   "Exception in ~SLOptixRaytracer: %s",
                   e.what());
    }
}
//-----------------------------------------------------------------------------
void SLOptixPathtracer::setupOptix()
{
    _cameraModule  = createModule("SLOptixPathtracerCamera.cu");
    _shadingModule = createModule("SLOptixPathtracerShading.cu");

    OptixProgramGroupDesc sample_raygen_desc    = {};
    sample_raygen_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    sample_raygen_desc.raygen.module            = _cameraModule;
    sample_raygen_desc.raygen.entryFunctionName = "__raygen__sample_camera";
    _pinhole_raygen_prog_group                  = createProgram(sample_raygen_desc);

    OptixProgramGroupDesc sample_miss_desc  = {};
    sample_miss_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    sample_miss_desc.miss.module            = _shadingModule;
    sample_miss_desc.miss.entryFunctionName = "__miss__sample";
    _radiance_miss_group                    = createProgram(sample_miss_desc);

    OptixProgramGroupDesc sample_hitgroup_desc        = {};
    sample_hitgroup_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sample_hitgroup_desc.hitgroup.moduleAH            = _shadingModule;
    sample_hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    sample_hitgroup_desc.hitgroup.moduleCH            = _shadingModule;
    sample_hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    _radiance_hit_group                               = createProgram(sample_hitgroup_desc);

    OptixProgramGroupDesc occlusion_miss_desc  = {};
    occlusion_miss_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    occlusion_miss_desc.miss.module            = nullptr;
    occlusion_miss_desc.miss.entryFunctionName = nullptr;
    _occlusion_miss_group                      = createProgram(occlusion_miss_desc);

    OptixProgramGroupDesc occlusion_hitgroup_desc        = {};
    occlusion_hitgroup_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_hitgroup_desc.hitgroup.moduleAH            = nullptr;
    occlusion_hitgroup_desc.hitgroup.entryFunctionNameAH = nullptr;
    occlusion_hitgroup_desc.hitgroup.moduleCH            = nullptr;
    occlusion_hitgroup_desc.hitgroup.entryFunctionNameCH = nullptr;
    _occlusion_hit_group                                 = createProgram(occlusion_hitgroup_desc);

    OptixProgramGroup path_tracer_program_groups[] = {
      _pinhole_raygen_prog_group,
      _radiance_miss_group,
      _occlusion_miss_group,
      _radiance_hit_group,
      _occlusion_hit_group,
    };

    _pipeline = createPipeline(path_tracer_program_groups, 5);

    OptixDenoiserOptions denoiserOptions;
    denoiserOptions.inputKind   = OPTIX_DENOISER_INPUT_RGB;
    denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;

    OPTIX_CHECK(optixDenoiserCreate(SLOptix::context,
                                    &denoiserOptions,
                                    &_optixDenoiser));

    OPTIX_CHECK(optixDenoiserSetModel(_optixDenoiser,
                                      OPTIX_DENOISER_MODEL_KIND_LDR,
                                      nullptr,
                                      0));
}
//-----------------------------------------------------------------------------
void SLOptixPathtracer::setupScene(SLSceneView* sv, SLAssetManager* am)
{
    SLVMesh meshes = am->meshes();
    _sv            = sv;

    _imageBuffer.resize(_sv->scrW() * _sv->scrH() * sizeof(float4));
    _curandBuffer.resize(_sv->scrW() * _sv->scrH() * sizeof(curandState));

    _params.image     = reinterpret_cast<float4*>(_imageBuffer.devicePointer());
    _params.states    = reinterpret_cast<curandState*>(_curandBuffer.devicePointer());
    _params.width     = _sv->scrW();
    _params.height    = _sv->scrH();
    _params.max_depth = _maxDepth;
    _params.samples   = _samples;

    // Iterate over all meshes
    SLMesh::meshIndex = 0;
    for (auto mesh : meshes)
        mesh->createMeshAccelerationStructure();

    _sbtClassic = createShaderBindingTable(meshes, false);

    OPTIX_CHECK(optixDenoiserComputeMemoryResources(_optixDenoiser,
                                                    _sv->scrW(),
                                                    _sv->scrW(),
                                                    &_denoiserSizes));

    _denoserState.resize(_denoiserSizes.stateSizeInBytes);
    _scratch.resize(_denoiserSizes.recommendedScratchSizeInBytes);

    OPTIX_CHECK(optixDenoiserSetup(
      _optixDenoiser,
      SLOptix::stream,
      _sv->scrW(),
      _sv->scrH(),
      _denoserState.devicePointer(),
      _denoiserSizes.stateSizeInBytes,
      _scratch.devicePointer(),
      _denoiserSizes.recommendedScratchSizeInBytes));
}
//-----------------------------------------------------------------------------
void SLOptixPathtracer::updateScene(SLSceneView* sv)
{
    SLScene*  scene  = sv->s();
    SLCamera* camera = sv->camera();
    _sv              = sv;

    SLNode::instanceIndex = 0;
    scene->root3D()->createInstanceAccelerationStructureFlat();

    _params.handle = scene->root3D()->optixTraversableHandle();

    SLVec3f eye, u, v, w;
    camera->UVWFrame(eye, u, v, w);
    ortCamera cameraData{};
    cameraData.eye = make_float3(eye);
    cameraData.U   = make_float3(u);
    cameraData.V   = make_float3(v);
    cameraData.W   = make_float3(w);

    RayGenClassicSbtRecord rayGenSbtRecord;
    _rayGenClassicBuffer.download(&rayGenSbtRecord);
    OPTIX_CHECK(optixSbtRecordPackHeader(_pinhole_raygen_prog_group,
                                         &rayGenSbtRecord));
    rayGenSbtRecord.data = cameraData;
    _rayGenClassicBuffer.upload(&rayGenSbtRecord);

    _params.seed = (SLuint)time(nullptr);

    _paramsBuffer.upload(&_params);
}
//-----------------------------------------------------------------------------
SLbool SLOptixPathtracer::render()
{
    _renderSec = 0.0f; // reset time
    // Measure time
    double t1     = GlobalTimer::timeMS();
    double tStart = t1;
    _state        = rtBusy; // From here we state the RT as busy

                            // Todo: Bugfix needed for Optix needs some work for newer shader models
    //OPTIX_CHECK(
    optixLaunch(
      _pipeline,
      SLOptix::stream,
      _paramsBuffer.devicePointer(),
      _paramsBuffer.size(),
      &_sbtClassic,
      _sv->scrW(),
      _sv->scrH(),
      /*depth=*/1);
    //);
    CUDA_SYNC_CHECK(SLOptix::stream);

    _renderSec = (SLfloat)(GlobalTimer::timeMS() - tStart) / 1000;

    // Measure denoiser time
    double t2 = GlobalTimer::timeMS();

    if (_denoiserEnabled)
    {
        OptixImage2D optixImage2D;
        optixImage2D.data               = _imageBuffer.devicePointer();
        optixImage2D.width              = _sv->scrW();
        optixImage2D.height             = _sv->scrH();
        optixImage2D.rowStrideInBytes   = _sv->scrW() * sizeof(float4);
        optixImage2D.pixelStrideInBytes = 0;
        optixImage2D.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixDenoiserParams denoiserParams;
        denoiserParams.denoiseAlpha = 0;
        denoiserParams.blendFactor  = 0.0f;
        denoiserParams.hdrIntensity = 0;

        OPTIX_CHECK(optixDenoiserInvoke(
          _optixDenoiser,
          SLOptix::stream,
          &denoiserParams,
          _denoserState.devicePointer(),
          _denoiserSizes.stateSizeInBytes,
          &optixImage2D,
          1,
          0,
          0,
          &optixImage2D,
          _scratch.devicePointer(),
          _denoiserSizes.recommendedScratchSizeInBytes));
        CUDA_SYNC_CHECK(SLOptix::stream);
    }

    _denoiserMS = (SLfloat)(GlobalTimer::timeMS() - t2);

    _state = rtFinished;
    return true;
}
//-----------------------------------------------------------------------------
#endif // SL_HAS_OPTIX
