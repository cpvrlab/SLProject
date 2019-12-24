//#############################################################################
//  File:      SLOptixPathtracer.cpp
//  Author:    Nic Dorner
//  Date:      Dezember 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################
#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLApplication.h>
#include <SLSceneView.h>
#include <SLOptixPathtracer.h>

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

SLOptixPathtracer::SLOptixPathtracer() {
    name("OptiX path tracer");
}

SLOptixPathtracer::~SLOptixPathtracer() {
    SL_LOG("Destructor      : ~SLOptixPathtracer\n");

    OPTIX_CHECK( optixDenoiserDestroy(_optixDenoiser) );
}

void SLOptixPathtracer::setupOptix() {
    OptixDeviceContext context = SLApplication::context;

    _cameraModule   = _createModule("SLOptixPathtracerCamera.cu");
    _shadingModule  = _createModule("SLOptixPathtracerShading.cu");

    OptixProgramGroupDesc sample_raygen_prog_group_desc  = {};
    sample_raygen_prog_group_desc.kind                                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    sample_raygen_prog_group_desc.raygen.module                            = _cameraModule;
    sample_raygen_prog_group_desc.raygen.entryFunctionName                 = "__raygen__sample_camera";
    _sample_raygen_prog_group = _createProgram(sample_raygen_prog_group_desc);

    OptixProgramGroupDesc sample_miss_prog_group_desc = {};
    sample_miss_prog_group_desc.kind                              = OPTIX_PROGRAM_GROUP_KIND_MISS;
    sample_miss_prog_group_desc.miss.module                       = _shadingModule;
    sample_miss_prog_group_desc.miss.entryFunctionName            = "__miss__sample";
    _sample_miss_group = _createProgram(sample_miss_prog_group_desc);

    OptixProgramGroupDesc sample_hit_prog_group_desc = {};
    sample_hit_prog_group_desc.kind                              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sample_hit_prog_group_desc.hitgroup.moduleAH             = _shadingModule;
    sample_hit_prog_group_desc.hitgroup.entryFunctionNameAH  = "__anyhit__radiance";
    sample_hit_prog_group_desc.hitgroup.moduleCH             = _shadingModule;
    sample_hit_prog_group_desc.hitgroup.entryFunctionNameCH  = "__closesthit__radiance";
    _sample_hit_group = _createProgram(sample_hit_prog_group_desc);

    OptixProgramGroup path_tracer_program_groups[] = {
            _sample_raygen_prog_group,
            _sample_miss_group,
            _sample_hit_group
    };
    _path_tracer_pipeline       = _createPipeline(path_tracer_program_groups, 3);

    OptixDenoiserOptions denoiserOptions;
    denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB;
    denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;

    OPTIX_CHECK( optixDenoiserCreate(context, &denoiserOptions, &_optixDenoiser) );

    OPTIX_CHECK( optixDenoiserSetModel(
            _optixDenoiser,
            OPTIX_DENOISER_MODEL_KIND_LDR,
            nullptr, 0) );
}

OptixShaderBindingTable SLOptixPathtracer::_createShaderBindingTable(const SLVMesh &meshes) {
    SLCamera* camera = _sv->camera();

    OptixShaderBindingTable sbt = {};
    {
        // Setup ray generation records
        RayGenPathtracerSbtRecord rg_sbt;
        _rayGenPathtracerBuffer.alloc_and_upload(&rg_sbt, 1);

        // Setup miss records
        std::vector<MissSbtRecord> missRecords;

        MissSbtRecord radiance_ms_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( _sample_miss_group , &radiance_ms_sbt ) );
        radiance_ms_sbt.data.bg_color = make_float4(camera->background().colors()[0]);
        missRecords.push_back(radiance_ms_sbt);

        _missBuffer.alloc_and_upload(missRecords);

        // Setup hit records
        std::vector<HitSbtRecord> hitRecords;

        for(auto mesh : meshes) {
            HitSbtRecord sample_hg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( _sample_hit_group, &sample_hg_sbt ) );
            sample_hg_sbt.data = mesh->createHitData();
            hitRecords.push_back(sample_hg_sbt);

            hitRecords.push_back(sample_hg_sbt);
        }
        _hitBuffer.alloc_and_upload(hitRecords);

        sbt.raygenRecord                = _rayGenPathtracerBuffer.devicePointer();
        sbt.missRecordBase              = _missBuffer.devicePointer();
        sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
        sbt.missRecordCount             = missRecords.size();
        sbt.hitgroupRecordBase          = _hitBuffer.devicePointer();
        sbt.hitgroupRecordStrideInBytes = sizeof( HitSbtRecord );
        sbt.hitgroupRecordCount         = hitRecords.size();
    }

    return sbt;
}

void SLOptixPathtracer::setupScene(SLSceneView *sv) {
    SLScene* scene = SLApplication::scene;
    SLVMesh meshes = scene->meshes();
    _sv = sv;

    _imageBuffer.resize(_sv->scrW() * _sv->scrH() * sizeof(uchar4));
    _curandBuffer.resize(_sv->scrW() * _sv->scrH() * sizeof(curandState));
    _debugBuffer.resize(_sv->scrW() * _sv->scrH() * sizeof(float3));

    _params.image = reinterpret_cast<uchar4 *>(_imageBuffer.devicePointer());
    _params.states = reinterpret_cast<curandState *>(_curandBuffer.devicePointer());
    _params.debug = reinterpret_cast<float3 *>(_debugBuffer.devicePointer());
    _params.width = _sv->scrW();
    _params.height = _sv->scrH();
    _params.max_depth = _maxDepth;
    _params.samples = _samples;

    // Iterate over all meshes
    SLMesh::meshIndex = 0;
    for(auto mesh : meshes) {
        mesh->createMeshAccelerationStructure();
    }

    _sbtPathtracer = _createShaderBindingTable(meshes);

    OPTIX_CHECK( optixDenoiserComputeMemoryResources(_optixDenoiser, _sv->scrW(), _sv->scrW(), &_denoiserSizes) );
    _denoserState.resize(_denoiserSizes.stateSizeInBytes);
    _scratch.resize(_denoiserSizes.recommendedScratchSizeInBytes);
    OPTIX_CHECK( optixDenoiserSetup(
            _optixDenoiser,
            SLApplication::stream,
            _sv->scrW(),
            _sv->scrW(),
            _denoserState.devicePointer(),
            _denoiserSizes.stateSizeInBytes,
            _scratch.devicePointer(),
            _denoiserSizes.recommendedScratchSizeInBytes) );
}

void SLOptixPathtracer::updateScene(SLSceneView *sv) {
    SLScene* scene = SLApplication::scene;
    SLCamera* camera = sv->camera();
    _sv = sv;

    SLNode::instanceIndex = 0;
//    scene->root3D()->createInstanceAccelerationStructureTree();
    scene->root3D()->createInstanceAccelerationStructureFlat();

    _params.handle = scene->root3D()->optixTraversableHandle();

    SLVec3f eye, u, v, w;
    camera->UVWFrame(eye, u, v, w);
    CameraData cameraData{};
    cameraData.eye = make_float3(eye);
    cameraData.U = make_float3(u);
    cameraData.V = make_float3(v);
    cameraData.W = make_float3(w);

    RayGenPathtracerSbtRecord rayGenSbtRecord;
    _rayGenPathtracerBuffer.download(&rayGenSbtRecord);
    OPTIX_CHECK( optixSbtRecordPackHeader(_sample_raygen_prog_group, &rayGenSbtRecord ) );
    rayGenSbtRecord.data = cameraData;
    _rayGenPathtracerBuffer.upload(&rayGenSbtRecord);

    _params.seed = time(NULL);

    _paramsBuffer.upload(&_params);
}

SLbool SLOptixPathtracer::render() {
    OPTIX_CHECK(optixLaunch(
            _path_tracer_pipeline,
            SLApplication::stream,
            _paramsBuffer.devicePointer(),
            _paramsBuffer.size(),
            &_sbtPathtracer,
            _sv->scrW(),
            _sv->scrH(),
            /*depth=*/1));
    CUDA_SYNC_CHECK(SLApplication::stream);

    OptixImage2D optixImage2D;
    optixImage2D.data = _imageBuffer.devicePointer();
    optixImage2D.width = _sv->scrW();
    optixImage2D.height = _sv->scrH();
    optixImage2D.rowStrideInBytes = 0;
    optixImage2D.pixelStrideInBytes = 0;
    optixImage2D.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = 0;
    denoiserParams.hdrIntensity = 0;

    OPTIX_CHECK( optixDenoiserInvoke(
            _optixDenoiser,
            SLApplication::stream,
            &denoiserParams,
            _denoserState.devicePointer(),
            _denoiserSizes.stateSizeInBytes,
            &optixImage2D,
            1,
            0,
            0,
            &optixImage2D,
            _scratch.devicePointer(),
            _denoiserSizes.recommendedScratchSizeInBytes) );

    return true;
}

void SLOptixPathtracer::renderImage() {
    SLOptixRaytracer::renderImage();
}

