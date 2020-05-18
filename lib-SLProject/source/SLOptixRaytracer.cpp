//#############################################################################
//  File:      SLOptixRaytracer.cpp
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    include <stdafx.h> // Must be the 1st include followed by  an empty line
#    include <SLApplication.h>
#    include <SLProjectScene.h>
#    include <SLLightRect.h>
#    include <SLSceneView.h>
#    include <SLOptix.h>
#    include <SLOptixRaytracer.h>
#    include <SLOptixDefinitions.h>
#    include <optix.h>
#    include <utility>
#    include <SLOptixHelper.h>
#    include <SLLine.h>
#    include <GlobalTimer.h>

//-----------------------------------------------------------------------------
SLOptixRaytracer::SLOptixRaytracer()
  : SLRaytracer()
{
    name("OptiX ray tracer");
    _params = {};
    _paramsBuffer.alloc(sizeof(ortParams));
    initCompileOptions();
}
//-----------------------------------------------------------------------------
SLOptixRaytracer::~SLOptixRaytracer()
{
    SL_LOG("Destructor      : ~SLOptixRaytracer");

    try
    {
        OPTIX_CHECK(optixPipelineDestroy(_pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(_radiance_hit_group));
        OPTIX_CHECK(optixProgramGroupDestroy(_occlusion_hit_group));
        OPTIX_CHECK(optixProgramGroupDestroy(_radiance_miss_group));
        OPTIX_CHECK(optixProgramGroupDestroy(_occlusion_miss_group));
        OPTIX_CHECK(optixProgramGroupDestroy(_pinhole_raygen_prog_group));
        OPTIX_CHECK(optixModuleDestroy(_cameraModule));
        OPTIX_CHECK(optixModuleDestroy(_shadingModule));
        OPTIX_CHECK(optixModuleDestroy(_traceModule));
    }
    catch (exception e)
    {
        Utils::log("SLProject",
                   "Exception in ~SLOptixRaytracer: %s",
                   e.what());
    }
}
//-----------------------------------------------------------------------------
void SLOptixRaytracer::initCompileOptions()
{
    // Set compile options for modules and pipelines
    _module_compile_options                  = {};
    _module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#    ifdef NDEBUG
    _module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    _module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    _pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#    else
    _module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    _module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    _pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_USER;
#    endif

    _pipeline_compile_options.usesMotionBlur = false;
    //_pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    _pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    _pipeline_compile_options.numPayloadValues                 = 7;
    _pipeline_compile_options.numAttributeValues               = 2;
    _pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
}
//-----------------------------------------------------------------------------
void SLOptixRaytracer::setupOptix()
{
    _cameraModule  = _createModule("SLOptixRaytracerCamera.cu");
    _shadingModule = _createModule("SLOptixRaytracerShading.cu");
    _traceModule   = _createModule("SLOptixTrace.cu");

    OptixProgramGroupDesc pinhole_raygen_desc    = {};
    pinhole_raygen_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pinhole_raygen_desc.raygen.module            = _cameraModule;
    pinhole_raygen_desc.raygen.entryFunctionName = "__raygen__pinhole_camera";
    _pinhole_raygen_prog_group                   = _createProgram(pinhole_raygen_desc);

    OptixProgramGroupDesc lens_raygen_desc    = {};
    lens_raygen_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    lens_raygen_desc.raygen.module            = _cameraModule;
    lens_raygen_desc.raygen.entryFunctionName = "__raygen__lens_camera";
    _lens_raygen_prog_group                   = _createProgram(lens_raygen_desc);

    OptixProgramGroupDesc orthographic_raygen_desc    = {};
    orthographic_raygen_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    orthographic_raygen_desc.raygen.module            = _cameraModule;
    orthographic_raygen_desc.raygen.entryFunctionName = "__raygen__orthographic_camera";
    _orthographic_raygen_prog_group                   = _createProgram(orthographic_raygen_desc);

    OptixProgramGroupDesc radiance_miss_desc  = {};
    radiance_miss_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    radiance_miss_desc.miss.module            = _shadingModule;
    radiance_miss_desc.miss.entryFunctionName = "__miss__radiance";
    _radiance_miss_group                      = _createProgram(radiance_miss_desc);

    OptixProgramGroupDesc occlusion_miss_desc  = {};
    occlusion_miss_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    occlusion_miss_desc.miss.module            = _shadingModule;
    occlusion_miss_desc.miss.entryFunctionName = "__miss__occlusion";
    _occlusion_miss_group                      = _createProgram(occlusion_miss_desc);

    OptixProgramGroupDesc radiance_hitgroup_desc        = {};
    radiance_hitgroup_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_hitgroup_desc.hitgroup.moduleAH            = _shadingModule;
    radiance_hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    radiance_hitgroup_desc.hitgroup.moduleCH            = _shadingModule;
    radiance_hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    _radiance_hit_group                                 = _createProgram(radiance_hitgroup_desc);

    OptixProgramGroupDesc radiance_hitgroup_line_desc        = {};
    radiance_hitgroup_line_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_hitgroup_line_desc.hitgroup.moduleIS            = _traceModule;
    radiance_hitgroup_line_desc.hitgroup.entryFunctionNameIS = "__intersection__line";
    radiance_hitgroup_line_desc.hitgroup.moduleAH            = _traceModule;
    radiance_hitgroup_line_desc.hitgroup.entryFunctionNameAH = "__anyhit__line_radiance";
    radiance_hitgroup_line_desc.hitgroup.moduleCH            = _traceModule;
    radiance_hitgroup_line_desc.hitgroup.entryFunctionNameCH = "__closesthit__line_radiance";
    _radiance_line_hit_group                                 = _createProgram(radiance_hitgroup_line_desc);

    OptixProgramGroupDesc occlusion_hitgroup_desc        = {};
    occlusion_hitgroup_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_hitgroup_desc.hitgroup.moduleAH            = _shadingModule;
    occlusion_hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__occlusion";
    occlusion_hitgroup_desc.hitgroup.moduleCH            = nullptr;
    occlusion_hitgroup_desc.hitgroup.entryFunctionNameCH = nullptr;
    _occlusion_hit_group                                 = _createProgram(occlusion_hitgroup_desc);

    OptixProgramGroupDesc occlusion_hitgroup_line_desc        = {};
    occlusion_hitgroup_line_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_hitgroup_line_desc.hitgroup.moduleIS            = _traceModule;
    occlusion_hitgroup_line_desc.hitgroup.entryFunctionNameIS = "__intersection__line";
    occlusion_hitgroup_line_desc.hitgroup.moduleAH            = _traceModule;
    occlusion_hitgroup_line_desc.hitgroup.entryFunctionNameAH = "__anyhit__line_occlusion";
    occlusion_hitgroup_line_desc.hitgroup.moduleCH            = nullptr;
    occlusion_hitgroup_line_desc.hitgroup.entryFunctionNameCH = nullptr;
    _occlusion_line_hit_group                                 = _createProgram(occlusion_hitgroup_line_desc);

    OptixProgramGroup program_groups[] = {
      _pinhole_raygen_prog_group,
      _radiance_miss_group,
      _occlusion_miss_group,
      _radiance_hit_group,
      _radiance_line_hit_group,
      _occlusion_hit_group,
      _occlusion_line_hit_group,
    };
    _pipeline = _createPipeline(program_groups, 7);
}
//-----------------------------------------------------------------------------
OptixModule SLOptixRaytracer::_createModule(string filename)
{
    OptixModule module = nullptr;
    {
        const string ptx = getPtxStringFromFile(std::move(filename));
        char         log[2048];
        size_t       sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
          SLOptix::context,
          &_module_compile_options,
          &_pipeline_compile_options,
          ptx.c_str(),
          ptx.size(),
          log,
          &sizeof_log,
          &module));
    }
    return module;
}
//-----------------------------------------------------------------------------
OptixProgramGroup SLOptixRaytracer::_createProgram(OptixProgramGroupDesc desc)
{
    OptixProgramGroup        program_group         = {};
    OptixProgramGroupOptions program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      SLOptix::context,
      &desc,
      1, // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &program_group));

    return program_group;
}
//-----------------------------------------------------------------------------
OptixPipeline SLOptixRaytracer::_createPipeline(OptixProgramGroup* program_groups,
                                                unsigned int       numProgramGroups)
{
    OptixPipeline            pipeline;
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = _maxDepth;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipeline_link_options.overrideUsesMotionBlur   = false;

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
      SLOptix::context,
      &_pipeline_compile_options,
      &pipeline_link_options,
      program_groups,
      numProgramGroups,
      log,
      &sizeof_log,
      &pipeline));

    return pipeline;
}
//-----------------------------------------------------------------------------
OptixShaderBindingTable SLOptixRaytracer::_createShaderBindingTable(const SLVMesh& meshes,
                                                                    const bool     doDistributed)
{
    SLCamera* camera = _sv->camera();

    OptixShaderBindingTable sbt = {};
    {
        // Setup ray generation records
        if (doDistributed)
        {
            RayGenDistributedSbtRecord rg_sbt;
            _rayGenDistributedBuffer.alloc_and_upload(&rg_sbt, 1);
        }
        else
        {
            RayGenClassicSbtRecord rg_sbt;
            _rayGenClassicBuffer.alloc_and_upload(&rg_sbt, 1);
        }

        // Setup miss records
        vector<MissSbtRecord> missRecords;

        MissSbtRecord radiance_ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(_radiance_miss_group, &radiance_ms_sbt));
        radiance_ms_sbt.data.bg_color = make_float4(camera->background().colors()[0]);
        missRecords.push_back(radiance_ms_sbt);

        MissSbtRecord occlusion_ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(_occlusion_miss_group, &occlusion_ms_sbt));
        missRecords.push_back(occlusion_ms_sbt);

        _missBuffer.alloc_and_upload(missRecords);

        // Setup hit records
        vector<HitSbtRecord> hitRecords;

        for (auto mesh : meshes)
        {
            OptixProgramGroup hitgroup_radicance = _radiance_hit_group;
            OptixProgramGroup hitgroup_occlusion = _occlusion_hit_group;
            if (mesh->name() == "line")
            {
                hitgroup_radicance = _radiance_line_hit_group;
                hitgroup_occlusion = _occlusion_line_hit_group;
            }
            HitSbtRecord radiance_hg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_radicance, &radiance_hg_sbt));
            radiance_hg_sbt.data = mesh->createHitData();
            hitRecords.push_back(radiance_hg_sbt);

            HitSbtRecord occlusion_hg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_occlusion, &occlusion_hg_sbt));
            occlusion_hg_sbt.data.material.kt             = mesh->mat()->kt();
            occlusion_hg_sbt.data.material.emissive_color = make_float4(mesh->mat()->emissive());
            hitRecords.push_back(occlusion_hg_sbt);
        }
        _hitBuffer.alloc_and_upload(hitRecords);

        if (doDistributed)
        {
            sbt.raygenRecord = _rayGenDistributedBuffer.devicePointer();
        }
        else
        {
            sbt.raygenRecord = _rayGenClassicBuffer.devicePointer();
        }
        sbt.missRecordBase              = _missBuffer.devicePointer();
        sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
        sbt.missRecordCount             = RAY_TYPE_COUNT;
        sbt.hitgroupRecordBase          = _hitBuffer.devicePointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitSbtRecord);
        sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * (SLuint)meshes.size();
    }

    return sbt;
}
//-----------------------------------------------------------------------------
void SLOptixRaytracer::setupScene(SLSceneView* sv)
{
    SLAssetManager* am     = (SLAssetManager*)SLApplication::scene;
    SLVMesh         meshes = am->meshes();
    _sv                    = sv;

    _imageBuffer.resize(_sv->scrW() * _sv->scrH() * sizeof(float4));
    _lineBuffer.resize(_sv->scrW() * _sv->scrH() * _maxDepth * 2 * sizeof(ortRay));

    _params.image     = reinterpret_cast<float4*>(_imageBuffer.devicePointer());
    _params.rays      = reinterpret_cast<ortRay*>(_lineBuffer.devicePointer());
    _params.width     = _sv->scrW();
    _params.height    = _sv->scrH();
    _params.max_depth = _maxDepth;

    // Iterate over all meshes
    SLMesh::meshIndex = 0;
    for (auto mesh : meshes)
    {
        mesh->createMeshAccelerationStructure();
    }

    _sbtClassic     = _createShaderBindingTable(meshes, false);
    _sbtDistributed = _createShaderBindingTable(meshes, true);
}
//-----------------------------------------------------------------------------
void SLOptixRaytracer::updateScene(SLSceneView* sv)
{
    SLScene*  scene  = SLApplication::scene;
    SLCamera* camera = sv->camera();
    _sv              = sv;

    SLNode::instanceIndex = 0;
    //    scene->root3D()->createInstanceAccelerationStructureTree();
    scene->root3D()->createInstanceAccelerationStructureFlat();

    _params.handle = scene->root3D()->optixTraversableHandle();

    SLVec3f eye, u, v, w;
    camera->UVWFrame(eye, u, v, w);
    ortCamera cameraData{};
    cameraData.eye = make_float3(eye);
    cameraData.U   = make_float3(u);
    cameraData.V   = make_float3(v);
    cameraData.W   = make_float3(w);

    if (doDistributed())
    {
        RayGenDistributedSbtRecord rayGenSbtRecord;
        _rayGenDistributedBuffer.download(&rayGenSbtRecord);
        OPTIX_CHECK(optixSbtRecordPackHeader(_lens_raygen_prog_group, &rayGenSbtRecord));
        rayGenSbtRecord.data.lensDiameter     = camera->lensDiameter();
        rayGenSbtRecord.data.samples.samplesX = camera->lensSamples()->samplesX();
        rayGenSbtRecord.data.samples.samplesY = camera->lensSamples()->samplesY();
        rayGenSbtRecord.data.camera           = cameraData;
        _rayGenDistributedBuffer.upload(&rayGenSbtRecord);
    }
    else
    {
        RayGenClassicSbtRecord rayGenSbtRecord;
        _rayGenClassicBuffer.download(&rayGenSbtRecord);
        if (camera->projection() == P_monoPerspective)
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(_pinhole_raygen_prog_group, &rayGenSbtRecord));
        }
        else
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(_orthographic_raygen_prog_group, &rayGenSbtRecord));
        }
        rayGenSbtRecord.data = cameraData;
        _rayGenClassicBuffer.upload(&rayGenSbtRecord);
    }

    vector<ortLight> lights;
    _lightBuffer.free();
    unsigned int light_count = 0;
    for (auto light : scene->lights())
    {
        if (light->isOn())
        {
            lights.push_back(light->optixLight(doDistributed()));
            light_count++;
        }
    }
    _lightBuffer.alloc_and_upload(lights);
    _params.lights             = reinterpret_cast<ortLight*>(_lightBuffer.devicePointer());
    _params.numLights          = light_count;
    _params.globalAmbientColor = make_float4(scene->globalAmbiLight());

    _paramsBuffer.upload(&_params);
}
//-----------------------------------------------------------------------------
SLbool SLOptixRaytracer::renderClassic()
{
    _state      = rtBusy; // From here we state the RT as busy
    _progressPC = 0;      // % rendered
    _renderSec  = 0.0f;   // reset time
    // Measure time
    double t1     = GlobalTimer::timeS();
    double tStart = t1;

    OPTIX_CHECK(optixLaunch(
      _pipeline,
      SLOptix::stream,
      _paramsBuffer.devicePointer(),
      _paramsBuffer.size(),
      &_sbtClassic,
      _sv->scrW(),
      _sv->scrH(),
      /*depth=*/1));
    CUDA_SYNC_CHECK(SLOptix::stream);

    _renderSec = (SLfloat)(GlobalTimer::timeS() - tStart);

    return true;
}
//-----------------------------------------------------------------------------
SLbool SLOptixRaytracer::renderDistrib()
{
    _renderSec = 0.0f; // reset time
    // Measure time
    double t1     = GlobalTimer::timeS();
    double tStart = t1;

    OPTIX_CHECK(optixLaunch(
      _pipeline,
      SLOptix::stream,
      _paramsBuffer.devicePointer(),
      _paramsBuffer.size(),
      &_sbtDistributed,
      _sv->scrW(),
      _sv->scrH(),
      /*depth=*/1));
    CUDA_SYNC_CHECK(SLOptix::stream);

    _renderSec = (SLfloat)(GlobalTimer::timeS() - tStart);

    return true;
}
//-----------------------------------------------------------------------------
void SLOptixRaytracer::prepareImage()
{
    // Create the image for the first time
    if (_images.empty())
        _images.push_back(new CVImage(_sv->scrW(), _sv->scrH(), PF_rgb, "Optix Raytracer"));

    // Allocate image of the inherited texture class
    if (_sv->scrW() != (SLint)_images[0]->width() ||
        _sv->scrH() != (SLint)_images[0]->height())
    {
        // Delete the OpenGL Texture if it already exists
        if (_texID)
        {
            //if (_cudaGraphicsResource) {
            //    CUDA_CHECK( cuGraphicsUnregisterResource(_cudaGraphicsResource) );
            //    _cudaGraphicsResource = nullptr;
            //}
            glDeleteTextures(1, &_texID);
            _texID = 0;
        }

        _images[0]->allocate(_sv->scrW(), _sv->scrH(), PF_rgb);
    }
}
//-----------------------------------------------------------------------------
void SLOptixRaytracer::renderImage()
{
    prepareImage(); // Setup image & precalculations
    SLGLTexture::bindActive(0);

    CUarray texture_ptr;
    CUDA_CHECK(cuGraphicsMapResources(1, &_cudaGraphicsResource, SLOptix::stream));
    CUDA_CHECK(cuGraphicsSubResourceGetMappedArray(&texture_ptr, _cudaGraphicsResource, 0, 0));

    CUDA_ARRAY_DESCRIPTOR des;
    cuArrayGetDescriptor(&des, texture_ptr);
    CUDA_MEMCPY2D memcpy2D;
    memcpy2D.srcDevice     = _imageBuffer.devicePointer();
    memcpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy2D.srcXInBytes   = 0;
    memcpy2D.srcY          = 0;
    memcpy2D.srcPitch      = 0;
    memcpy2D.dstArray      = texture_ptr;
    memcpy2D.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy2D.dstXInBytes   = 0;
    memcpy2D.dstY          = 0;
    memcpy2D.dstPitch      = 0;
    memcpy2D.WidthInBytes  = des.Width * des.NumChannels * sizeof(float);
    memcpy2D.Height        = des.Height;
    CUDA_CHECK(cuMemcpy2D(&memcpy2D));

    CUDA_CHECK(cuGraphicsUnmapResources(1, &_cudaGraphicsResource, SLOptix::stream));

    SLfloat w = (SLfloat)_sv->scrW();
    SLfloat h = (SLfloat)_sv->scrH();
    if (Utils::abs(_images[0]->width() - w) > 0.0001f) return;
    if (Utils::abs(_images[0]->height() - h) > 0.0001f) return;

    // Set orthographic projection with the size of the window
    SLGLState* stateGL = SLGLState::instance();
    stateGL->projectionMatrix.ortho(0.0f, w, 0.0f, h, -1.0f, 0.0f);
    stateGL->modelViewMatrix.identity();
    stateGL->clearColorBuffer();
    stateGL->depthTest(false);
    stateGL->multiSample(false);
    stateGL->polygonLine(false);

    drawSprite(false, 0.0f, 0.0f, w, h);

    stateGL->depthTest(true);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
void SLOptixRaytracer::saveImage()
{
    float4* image = static_cast<float4*>(malloc(_imageBuffer.size()));
    _imageBuffer.download(image);

    for (int i = 0; i < _sv->scrH(); i++)
    {
        for (int j = 0; j < _sv->scrW(); j++)
        {
            float4 pixel = image[i * _sv->scrW() + j];
            _images[0]->setPixeli(j, i, CVVec4f(pixel.x, pixel.y, pixel.z));
        }
    }

    SLRaytracer::saveImage();
}
//-----------------------------------------------------------------------------
/*
void SLOptixRaytracer::drawRay(unsigned int x, unsigned int y)
{
    SLAssetManager* assetMngr = (SLAssetManager*)SLApplication::scene;

    y = _sv->scrH() - y;

    Ray* rays = static_cast<Ray*>(malloc(_lineBuffer.size()));
    _lineBuffer.download(rays);

    for (int i = 0; i < _maxDepth * 2; i++)
    {
        Ray ray = rays[(y * _sv->scrW() + x) * _maxDepth * 2 + i];

        auto* mat  = new SLMaterial(assetMngr,
                                   "mat",
                                   SLCol4f(ray.color.x, ray.color.y, ray.color.z),
                                   SLCol4f::BLACK,
                                   0);
        auto* line = new SLNode(new SLLine(assetMngr,
                                           SLVec3f(ray.line.p1.x,
                                                   ray.line.p1.y,
                                                   ray.line.p1.z),
                                           SLVec3f(ray.line.p2.x,
                                                   ray.line.p2.y,
                                                   ray.line.p2.z),
                                           mat),
                                "line");
        _sv->s().root3D()->addChild(line);
    }
    setupScene(_sv);
}
//-----------------------------------------------------------------------------
void SLOptixRaytracer::removeRays()
{
    SLScene* scene = SLApplication::scene;

    while (SLApplication::scene->root3D()->deleteChild("line")) {}
}
*/
//-----------------------------------------------------------------------------
#endif
