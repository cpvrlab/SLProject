//#############################################################################
//  File:      SLDemoGui.cpp
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLDemoGui.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLInterface.h>
#include <SLImporter.h>
#include <SLCVCapture.h>
#include <imgui.h>

//-----------------------------------------------------------------------------
SLbool SLDemoGui::showAbout = false;
SLbool SLDemoGui::showHelp = false;
SLbool SLDemoGui::showHelpCalibration = false;
SLbool SLDemoGui::showCredits = false;
SLbool SLDemoGui::showStatsTiming = false;
SLbool SLDemoGui::showStatsRenderer = false;
SLbool SLDemoGui::showStatsScene = false;
SLbool SLDemoGui::showStatsVideo = false;
//-----------------------------------------------------------------------------
void SLDemoGui::buildDemoGui(SLScene* s, SLSceneView* sv)
{
    buildMenuBar(s, sv);

    if (showAbout)
    {
        ImGui::Begin("About SLProject", &showAbout, ImVec2(400,0));
        ImGui::Text("Version %s", SL::version.c_str());
        ImGui::Separator();
        ImGui::TextWrapped(SL::infoAbout.c_str());
        ImGui::End();
    }

    if (showHelp)
    {
        ImGui::Begin("About SLProject Interaction", &showHelp, ImVec2(400,0));
        ImGui::TextWrapped(SL::infoHelp.c_str());
        ImGui::End();
    }

    if (showHelpCalibration)
    {
        ImGui::Begin("About Camera Calibration", &showHelpCalibration, ImVec2(400,0));
        ImGui::TextWrapped(SL::infoCalibrate.c_str());
        ImGui::End();
    }

    if (showCredits)
    {
        ImGui::Begin("Credits for all Helpers", &showCredits, ImVec2(400,0));
        ImGui::TextWrapped(SL::infoCredits.c_str());
        ImGui::End();
    }

    if (showStatsTiming)
    {
        SLRenderType rType = sv->renderType();
        SLCamera* cam = sv->camera();
        SLfloat ft = s->frameTimesMS().average()*100.0f;
        SLchar m[2550];   // message character array
        m[0]=0;           // set zero length

        if (rType == RT_gl)
        {
            SLfloat updateTimePC    = SL_clamp(s->updateTimesMS().average()   / ft, 0.0f,100.0f);
            SLfloat trackingTimePC  = SL_clamp(s->trackingTimesMS().average() / ft, 0.0f,100.0f);
            SLfloat cullTimePC      = SL_clamp(s->cullTimesMS().average()     / ft, 0.0f,100.0f);
            SLfloat draw3DTimePC    = SL_clamp(s->draw3DTimesMS().average()   / ft, 0.0f,100.0f);
            SLfloat draw2DTimePC    = SL_clamp(s->draw2DTimesMS().average()   / ft, 0.0f,100.0f);
            SLfloat captureTimePC   = SL_clamp(s->captureTimesMS().average()  / ft, 0.0f,100.0f);

            sprintf(m+strlen(m), "FPS: %4.1f  (Size: %d x %d, DPI: %d)\n", s->fps(), sv->scrW(), sv->scrH(), SL::dpi);
            sprintf(m+strlen(m), "Frame Time : %4.1f ms (100%%)\n", s->frameTimesMS().average());
            sprintf(m+strlen(m), "Update Time : %4.1f ms (%0.0f%%)\n", s->updateTimesMS().average(), updateTimePC);
            sprintf(m+strlen(m), "> Tracking Time: %4.1f ms (%0.0f%%)\n", s->trackingTimesMS().average(), trackingTimePC);
            sprintf(m+strlen(m), "Culling Time : %4.1f ms (%0.0f%%)\n", s->cullTimesMS().average(), cullTimePC);
            sprintf(m+strlen(m), "Draw Time 3D: %4.1f ms (%0.0f%%)\n", s->draw3DTimesMS().average(), draw3DTimePC);
            sprintf(m+strlen(m), "Draw Time 2D: %4.1f ms (%0.0f%%)\n", s->draw2DTimesMS().average(), draw2DTimePC);
            #ifdef SL_GLSL
            sprintf(m+strlen(m), "Capture Time: %4.1f ms\n", s->captureTimesMS().average());
            #else
            sprintf(m+strlen(m), "Capture Time: %4.1f ms (%0.0f%%)\n", s->captureTimesMS().average(), captureTimePC);
            #endif
            sprintf(m+strlen(m), "NO. of drawcalls: %d\n", SLGLVertexArray::totalDrawCalls);

            ImGui::Begin("Timing for OpenGL Renderer", &showStatsTiming, ImVec2(400,0));
            ImGui::TextWrapped(m);
            ImGui::End();

        } else
        if (rType == RT_rt)
        {
            SLRaytracer* rt = sv->raytracer();
            SLint primaries = sv->scrW() * sv->scrH();
            SLuint total = primaries + SLRay::reflectedRays + SLRay::subsampledRays + SLRay::refractedRays + SLRay::shadowRays;
            SLfloat rpms = rt->renderSec() ? total/rt->renderSec()/1000.0f : 0.0f;
            sprintf(m+strlen(m), "Timing -------------------------------------\\n");
            sprintf(m+strlen(m), "Scene: %s\\n", s->name().c_str());
            sprintf(m+strlen(m), "Time per frame: %4.2f sec.  (Size: %d x %d)\\n", rt->renderSec(), sv->scrW(), sv->scrH());
            sprintf(m+strlen(m), "fov: %4.2f\\n", cam->fov());
            sprintf(m+strlen(m), "Focal dist. (f): %4.2f\\n", cam->focalDist());
            sprintf(m+strlen(m), "Rays per millisecond: %6.0f\\n", rpms);
            sprintf(m+strlen(m), "Threads: %d\\n", rt->numThreads());
        }
    }


}
//-----------------------------------------------------------------------------
void SLDemoGui::buildMenuBar(SLScene* s, SLSceneView* sv)
{
    SLCommand curS = SL::currentSceneID;
    SLRenderType rType = sv->renderType();

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::BeginMenu("Load Scene"))
            {
                if (ImGui::BeginMenu("General Scenes"))
                {
                    SLstring large1 = SLImporter::defaultPath + "PLY/xyzrgb_dragon.ply";
                    SLstring large2 = SLImporter::defaultPath + "PLY/mesh_zermatt.ply";
                    SLstring large3 = SLImporter::defaultPath + "PLY/switzerland.ply";
                    SLbool largeFileExists = SLFileSystem::fileExists(large1) ||
                                             SLFileSystem::fileExists(large2) ||
                                             SLFileSystem::fileExists(large3);

                    if (ImGui::MenuItem("Minimal Scene", 0, curS==C_sceneMinimal))
                        sv->onCommand(C_sceneMinimal);
                    if (ImGui::MenuItem("Figure Scene", 0, curS==C_sceneFigure))
                        sv->onCommand(C_sceneFigure);
                    if (ImGui::MenuItem("Large Model", 0, curS==C_sceneLargeModel, largeFileExists))
                        sv->onCommand(C_sceneLargeModel);
                    if (ImGui::MenuItem("Mesh Loader", 0, curS==C_sceneMeshLoad))
                        sv->onCommand(C_sceneMeshLoad);
                    if (ImGui::MenuItem("Texture Blending", 0, curS==C_sceneTextureBlend))
                        sv->onCommand(C_sceneTextureBlend);
                    if (ImGui::MenuItem("Texture Filters", 0, curS==C_sceneTextureFilter))
                        sv->onCommand(C_sceneTextureFilter);
                    if (ImGui::MenuItem("Frustum Culling", 0, curS==C_sceneFrustumCull))
                        sv->onCommand(C_sceneFrustumCull);
                    if (ImGui::MenuItem("Massive Data Scene", 0, curS==C_sceneMassiveData))
                        sv->onCommand(C_sceneMassiveData);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Shader"))
                {
                    if (ImGui::MenuItem("Per Vertex Blinn-Phong Lighting", 0, curS==C_sceneShaderPerVertexBlinn))
                        sv->onCommand(C_sceneShaderPerVertexBlinn);
                    if (ImGui::MenuItem("Per Pixel Blinn-Phing Lighting", 0, curS==C_sceneShaderPerPixelBlinn))
                        sv->onCommand(C_sceneShaderPerPixelBlinn);
                    if (ImGui::MenuItem("Per Pixel Cook-Torrance Lighting", 0, curS==C_sceneShaderPerPixelCookTorrance))
                        sv->onCommand(C_sceneShaderPerPixelCookTorrance);
                    if (ImGui::MenuItem("Per Vertex Wave", 0, curS==C_sceneShaderPerVertexWave))
                        sv->onCommand(C_sceneShaderPerVertexWave);
                    if (ImGui::MenuItem("Water", 0, curS==C_sceneShaderWater))
                        sv->onCommand(C_sceneShaderWater);
                    if (ImGui::MenuItem("Bump Mapping", 0, curS==C_sceneShaderBumpNormal))
                        sv->onCommand(C_sceneShaderBumpNormal);
                    if (ImGui::MenuItem("Parallax Mapping", 0, curS==C_sceneShaderBumpParallax))
                        sv->onCommand(C_sceneShaderBumpParallax);
                    if (ImGui::MenuItem("Glass Shader", 0, curS==C_sceneRevolver))
                        sv->onCommand(C_sceneRevolver);
                    if (ImGui::MenuItem("Earth Shader", 0, curS==C_sceneShaderEarth))
                        sv->onCommand(C_sceneShaderEarth);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Animation"))
                {
                    if (ImGui::MenuItem("Mass Animation", 0, curS==C_sceneAnimationMass))
                        sv->onCommand(C_sceneAnimationMass);
                    if (ImGui::MenuItem("Astroboy Army", 0, curS==C_sceneAnimationArmy))
                        sv->onCommand(C_sceneAnimationArmy);
                    if (ImGui::MenuItem("Skeletal Animation", 0, curS==C_sceneAnimationSkeletal))
                        sv->onCommand(C_sceneAnimationSkeletal);
                    if (ImGui::MenuItem("Node Animation", 0, curS==C_sceneAnimationNode))
                        sv->onCommand(C_sceneAnimationNode);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Using Video"))
                {
                    if (ImGui::MenuItem("Track ArUco Marker (Main)", 0, curS==C_sceneVideoTrackArucoMain))
                        sv->onCommand(C_sceneVideoTrackArucoMain);
                    if (ImGui::MenuItem("Track ArUco Marker (Scnd)", 0, curS==C_sceneVideoTrackArucoScnd, SLCVCapture::hasSecondaryCamera))
                        sv->onCommand(C_sceneVideoTrackArucoScnd);
                    if (ImGui::MenuItem("Track Chessboard (Main)", 0, curS==C_sceneVideoTrackChessMain))
                        sv->onCommand(C_sceneVideoTrackChessMain);
                    if (ImGui::MenuItem("Track Chessboard (Scnd)", 0, curS==C_sceneVideoTrackChessScnd, SLCVCapture::hasSecondaryCamera))
                        sv->onCommand(C_sceneVideoTrackChessScnd);
                    if (ImGui::MenuItem("Texture from live video", 0, curS==C_sceneVideoTexture))
                        sv->onCommand(C_sceneVideoTexture);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Ray tracing"))
                {
                    if (ImGui::MenuItem("Spheres", 0, curS==C_sceneRTSpheres))
                        sv->onCommand(C_sceneRTSpheres);
                    if (ImGui::MenuItem("Muttenzer Box", 0, curS==C_sceneRTMuttenzerBox))
                        sv->onCommand(C_sceneRTMuttenzerBox);
                    if (ImGui::MenuItem("Soft Shadows", 0, curS==C_sceneRTSoftShadows))
                        sv->onCommand(C_sceneRTSoftShadows);
                    if (ImGui::MenuItem("Depth of Field", 0, curS==C_sceneRTDoF))
                        sv->onCommand(C_sceneRTDoF);
                    if (ImGui::MenuItem("Lens Test", 0, curS==C_sceneRTLens))
                        sv->onCommand(C_sceneRTLens);
                    if (ImGui::MenuItem("RT Test", 0, curS==C_sceneRTTest))
                        sv->onCommand(C_sceneRTTest);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Path tracing"))
                {
                    if (ImGui::MenuItem("Muttenzer Box", 0, curS==C_sceneRTMuttenzerBox))
                        sv->onCommand(C_sceneRTMuttenzerBox);

                    ImGui::EndMenu();
                }

                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::BeginMenu("Preferences"))
            {
                if (ImGui::BeginMenu("Rendering"))
                {
                    if (ImGui::MenuItem("Slow down on Idle", 0, sv->waitEvents()))
                        sv->onCommand(C_multiSampleToggle);

                    if (ImGui::MenuItem("Do Multi Sampling", 0, sv->doMultiSampling()))
                        sv->onCommand(C_waitEventsToggle);

                    if (ImGui::MenuItem("Do Frustum Culling", 0, sv->doFrustumCulling()))
                        sv->onCommand(C_frustCullToggle);

                    if (ImGui::MenuItem("Do Depth Test", 0, sv->doDepthTest()))
                        sv->onCommand(C_depthTestToggle);

                    if (ImGui::MenuItem("Animation off", 0, s->stopAnimations()))
                        sv->onCommand(C_animationToggle);

                    ImGui::EndMenu();
                }


                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Quit & Save"))
                sv->onCommand(C_quit);

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Renderer"))
        {
            if (ImGui::MenuItem("OpenGL", 0, rType==RT_gl))
                sv->onCommand(C_renderOpenGL);

            if (ImGui::MenuItem("Ray Tracing", 0, rType==RT_rt))
                sv->onCommand(C_rt5);

            if (ImGui::MenuItem("Path Tracing", 0, rType==RT_pt))
                sv->onCommand(C_pt10);

            ImGui::EndMenu();
        }

        if (rType == RT_gl)
        {
            if (ImGui::BeginMenu("Render Flags"))
            {
                if (ImGui::MenuItem("Wired Mesh", 0, sv->drawBits()->get(SL_DB_WIREMESH)))
                    sv->onCommand(C_wireMeshToggle);

                if (ImGui::MenuItem("Normals", 0, sv->drawBits()->get(SL_DB_NORMALS)))
                    sv->onCommand(C_normalsToggle);

                if (ImGui::MenuItem("Voxels", 0, sv->drawBits()->get(SL_DB_VOXELS)))
                    sv->onCommand(C_voxelsToggle);

                if (ImGui::MenuItem("Axis", 0, sv->drawBits()->get(SL_DB_AXIS)))
                    sv->onCommand(C_axisToggle);

                if (ImGui::MenuItem("Bounding Boxes", 0, sv->drawBits()->get(SL_DB_BBOX)))
                    sv->onCommand(C_bBoxToggle);

                if (ImGui::MenuItem("Skeleton", 0, sv->drawBits()->get(SL_DB_SKELETON)))
                    sv->onCommand(C_skeletonToggle);

                if (ImGui::MenuItem("Back Faces", 0, sv->drawBits()->get(SL_DB_CULLOFF)))
                    sv->onCommand(C_faceCullToggle);

                if (ImGui::MenuItem("Textures off", 0, sv->drawBits()->get(SL_DB_TEXOFF)))
                    sv->onCommand(C_textureToggle);

                if (ImGui::MenuItem("All Off"))
                    sv->drawBits()->allOff();

                if (ImGui::MenuItem("All on"))
                {
                    sv->drawBits()->on(SL_DB_WIREMESH);
                    sv->drawBits()->on(SL_DB_NORMALS);
                    sv->drawBits()->on(SL_DB_VOXELS);
                    sv->drawBits()->on(SL_DB_AXIS);
                    sv->drawBits()->on(SL_DB_BBOX);
                    sv->drawBits()->on(SL_DB_SKELETON);
                    sv->drawBits()->on(SL_DB_CULLOFF);
                    sv->drawBits()->on(SL_DB_TEXOFF);
                }

                ImGui::EndMenu();
            }
        }
        else if (rType == RT_rt)
        {
            if (ImGui::BeginMenu("RT Settings"))
            {
                SLRaytracer* rt = sv->raytracer();

                if (ImGui::MenuItem("Parallel distributed", 0, rt->distributed()))
                    sv->onCommand(C_rtDistributed);

                if (ImGui::MenuItem("Continuously", 0, rt->continuous()))
                    sv->onCommand(C_rtContinuously);

                if (ImGui::BeginMenu("Max. Depth"))
                {
                    if (ImGui::MenuItem("1", 0, rt->maxDepth()==1))
                        sv->onCommand(C_rt1);
                    if (ImGui::MenuItem("2", 0, rt->maxDepth()==2))
                        sv->onCommand(C_rt2);
                    if (ImGui::MenuItem("3", 0, rt->maxDepth()==3))
                        sv->onCommand(C_rt3);
                    if (ImGui::MenuItem("5", 0, rt->maxDepth()==5))
                        sv->onCommand(C_rt5);
                    if (ImGui::MenuItem("Max. Contribution", 0, rt->maxDepth()==0))
                        sv->onCommand(C_rt0);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Anti-Aliasing Sub Samples"))
                {
                    if (ImGui::MenuItem("Off", 0, rt->aaSamples()==1))
                        rt->aaSamples(1);
                    if (ImGui::MenuItem("3x3", 0, rt->aaSamples()==3))
                        rt->aaSamples(3);
                    if (ImGui::MenuItem("5x5", 0, rt->aaSamples()==5))
                        rt->aaSamples(5);
                    if (ImGui::MenuItem("7x7", 0, rt->aaSamples()==7))
                        rt->aaSamples(7);
                    if (ImGui::MenuItem("9x9", 0, rt->aaSamples()==9))
                        rt->aaSamples(9);

                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem("Save Rendered Image"))
                    sv->onCommand(C_rtSaveImage);

                ImGui::EndMenu();
            }
        }
        else if (rType == RT_pt)
        {
            if (ImGui::BeginMenu("PT Settings"))
            {
                SLPathtracer* pt = sv->pathtracer();

                if (ImGui::BeginMenu("NO. of Samples"))
                {
                    if (ImGui::MenuItem("1", 0, pt->aaSamples()==1))
                        sv->onCommand(C_pt1);
                    if (ImGui::MenuItem("10", 0, pt->aaSamples()==10))
                        sv->onCommand(C_pt10);
                    if (ImGui::MenuItem("100", 0, pt->aaSamples()==100))
                        sv->onCommand(C_pt100);
                    if (ImGui::MenuItem("1000", 0, pt->aaSamples()==1000))
                        sv->onCommand(C_pt1000);
                    if (ImGui::MenuItem("10000", 0, pt->aaSamples()==10000))
                        sv->onCommand(C_pt10000);

                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem("Save Rendered Image"))
                    sv->onCommand(C_ptSaveImage);

                ImGui::EndMenu();
            }
        }

        if (ImGui::BeginMenu("View"))
        {
            SLCamera* cam = sv->camera();
            SLProjection proj = cam->projection();

            if (ImGui::MenuItem("Reset"))
                sv->onCommand(C_camReset);

            if (s->numSceneCameras())
            {
                if (ImGui::MenuItem("Set next camera in Scene"))
                    sv->onCommand(C_camSetNextInScene);

                if (ImGui::MenuItem("Set SceneView Camera"))
                    sv->onCommand(C_camSetSceneViewCamera);
            }

            if (ImGui::BeginMenu("Projection"))
            {
                static SLfloat clipN = cam->clipNear();
                static SLfloat clipF = cam->clipFar();
                static SLfloat focalDist = cam->focalDist();
                static SLfloat fov = cam->fov();

                if (ImGui::MenuItem("Perspective", 0, proj==P_monoPerspective))
                    sv->onCommand(C_projPersp);

                if (ImGui::MenuItem("Orthographic", 0, proj==P_monoOrthographic))
                    sv->onCommand(C_projOrtho);

                if (ImGui::BeginMenu("Stereo"))
                {
                    for (SLint p=P_stereoSideBySide; p<=P_stereoColorYB; ++p)
                    {
                        SLstring pStr = SLCamera::projectionToStr((SLProjection)p);
                        if (ImGui::MenuItem(pStr.c_str(), 0, proj==(SLProjection)p))
                            sv->onCommand((SLCommand)(C_projPersp+p));
                    }

                    if (proj >=P_stereoSideBySide)
                    {
                        ImGui::Separator();
                        static SLfloat eyeSepar = cam->eyeSeparation();
                        if (ImGui::SliderFloat("Eye Sep.", &eyeSepar, 0.0f, focalDist/10.f))
                            cam->eyeSeparation(eyeSepar);
                    }

                    ImGui::EndMenu();
                }

                ImGui::Separator();

                if (ImGui::SliderFloat("FOV", &fov, 1.f, 179.f))
                    cam->fov(fov);

                if (ImGui::SliderFloat("Near Clip", &clipN, 0.001f, 10.f))
                    cam->clipNear(clipN);

                if (ImGui::SliderFloat("Far Clip",  &clipF, clipN, SL_min(clipF*1.1f,1000000.f)))
                    cam->clipFar(clipF);

                if (ImGui::SliderFloat("Focal Dist.", &focalDist, clipN, clipF))
                    cam->focalDist(focalDist);

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Animation"))
            {
                SLCamAnim ca = cam->camAnim();
                if (ImGui::MenuItem("Turntable Z up", 0, ca==CA_turntableZUp))
                    sv->onCommand(C_camAnimTurnZUp);

                if (ImGui::MenuItem("Turntable Y up", 0, ca==CA_turntableYUp))
                    sv->onCommand(C_camAnimTurnZUp);

                if (ImGui::MenuItem("Walk Z up", 0, ca==CA_walkingZUp))
                    sv->onCommand(C_camAnimWalkZUp);

                if (ImGui::MenuItem("Walk Y up", 0, ca==CA_walkingYUp))
                    sv->onCommand(C_camAnimWalkYUp);

                if (ca==CA_walkingZUp || ca==CA_walkingYUp)
                {
                    static SLfloat ms = cam->maxSpeed();
                    if (ImGui::SliderFloat("Walk Speed",  &ms, 0.01f, SL_min(ms*1.1f,10000.f)))
                        cam->maxSpeed(ms);
                }

                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Infos"))
        {
            ImGui::MenuItem("Stats on Timing", 0, &showStatsTiming);
            ImGui::MenuItem("Help on Interaction", 0, &showHelp);
            ImGui::MenuItem("Help on Calibration", 0, &showHelpCalibration);
            ImGui::MenuItem("About SLProject", 0, &showAbout);
            ImGui::MenuItem("Credits", 0, &showCredits);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}
//-----------------------------------------------------------------------------
