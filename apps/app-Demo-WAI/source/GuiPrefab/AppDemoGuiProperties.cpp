#include <imgui.h>
#include <imgui_internal.h>

#include <SLLightDirect.h>
#include <SLLightRect.h>
#include <SLLightSpot.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiProperties.h>
#include <SLTransferFunction.h>
#include <SLGLShader.h>
#include <Utils.h>
//-----------------------------------------------------------------------------
AppDemoGuiProperties::AppDemoGuiProperties(std::string name, bool* activator, ImFont* font)
  : AppDemoGuiInfosDialog(name, activator, font)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiProperties::buildInfos(SLScene* s, SLSceneView* sv)
{
    SLNode* singleNode       = s->singleNodeSelected();
    SLMesh* singleFullMesh   = s->singleMeshFullSelected();
    bool    partialSelection = !s->selectedMeshes().empty() && !s->selectedMeshes()[0]->IS32.empty();

    ImGui::PushFont(_font);

    if (singleNode && !partialSelection)
    {
        ImGui::Begin("Properties of Selection", _activator);

        if (ImGui::TreeNode("Single Node Properties"))
        {
            if (singleNode)
            {
                SLuint c = (SLuint)singleNode->children().size();
                SLuint m = singleNode->mesh() ? 1 : 0;
                ImGui::Text("Node Name       : %s", singleNode->name().c_str());
                ImGui::Text("No. of children : %u", c);
                ImGui::Text("No. of meshes   : %u", m);
                if (ImGui::TreeNode("Drawing Flags"))
                {
                    SLbool db;
                    db = singleNode->drawBit(SL_DB_HIDDEN);
                    if (ImGui::Checkbox("Hide", &db))
                        singleNode->drawBits()->set(SL_DB_HIDDEN, db);

                    db = singleNode->drawBit(SL_DB_MESHWIRED);
                    if (ImGui::Checkbox("Show wireframe", &db))
                        singleNode->drawBits()->set(SL_DB_MESHWIRED, db);

                    db = singleNode->drawBit(SL_DB_WITHEDGES);
                    if (ImGui::Checkbox("Show with hard edges", &db))
                        singleNode->drawBits()->set(SL_DB_ONLYEDGES, db);

                    db = singleNode->drawBit(SL_DB_ONLYEDGES);
                    if (ImGui::Checkbox("Show only hard edges", &db))
                        singleNode->drawBits()->set(SL_DB_WITHEDGES, db);

                    db = singleNode->drawBit(SL_DB_NORMALS);
                    if (ImGui::Checkbox("Show normals", &db))
                        singleNode->drawBits()->set(SL_DB_NORMALS, db);

                    db = singleNode->drawBit(SL_DB_VOXELS);
                    if (ImGui::Checkbox("Show voxels", &db))
                        singleNode->drawBits()->set(SL_DB_VOXELS, db);

                    db = singleNode->drawBit(SL_DB_BBOX);
                    if (ImGui::Checkbox("Show bounding boxes", &db))
                        singleNode->drawBits()->set(SL_DB_BBOX, db);

                    db = singleNode->drawBit(SL_DB_AXIS);
                    if (ImGui::Checkbox("Show axis", &db))
                        singleNode->drawBits()->set(SL_DB_AXIS, db);

                    db = singleNode->drawBit(SL_DB_CULLOFF);
                    if (ImGui::Checkbox("Show back faces", &db))
                        singleNode->drawBits()->set(SL_DB_CULLOFF, db);

                    ImGui::TreePop();
                }

                if (ImGui::TreeNode("Local Transform"))
                {
                    SLMat4f om(singleNode->om());
                    SLVec3f trn, rot, scl;
                    om.decompose(trn, rot, scl);
                    rot *= Utils::RAD2DEG;

                    ImGui::Text("Translation  : %s", trn.toString().c_str());
                    ImGui::Text("Rotation     : %s", rot.toString().c_str());
                    ImGui::Text("Scaling      : %s", scl.toString().c_str());
                    ImGui::TreePop();
                }

                // Show special camera properties
                if (typeid(*singleNode) == typeid(SLCamera))
                {
                    SLCamera* cam = (SLCamera*)singleNode;

                    if (ImGui::TreeNode("Camera"))
                    {
                        SLfloat clipN     = cam->clipNear();
                        SLfloat clipF     = cam->clipFar();
                        SLfloat focalDist = cam->focalDist();
                        SLfloat fov       = cam->fov();

                        const char* projections[] = {"Mono Perspective",
                                                     "Mono Orthographic",
                                                     "Stereo Side By Side",
                                                     "Stereo Side By Side Prop.",
                                                     "Stereo Side By Side Dist.",
                                                     "Stereo Line By Line",
                                                     "Stereo Column By Column",
                                                     "Stereo Pixel By Pixel",
                                                     "Stereo Color Red-Cyan",
                                                     "Stereo Color Red-Green",
                                                     "Stereo Color Red-Blue",
                                                     "Stereo Color Yellow-Blue"};

                        int proj = cam->projection();
                        if (ImGui::Combo("Projection", &proj, projections, IM_ARRAYSIZE(projections)))
                            cam->projection((SLProjection)proj);

                        if (cam->projection() > P_monoOrthographic)
                        {
                            SLfloat eyeSepar = cam->stereoEyeSeparation();
                            if (ImGui::SliderFloat("Eye Sep.", &eyeSepar, 0.0f, focalDist / 10.f))
                                cam->stereoEyeSeparation(eyeSepar);
                        }

                        if (ImGui::SliderFloat("FOV", &fov, 1.f, 179.f))
                            cam->fov(fov);

                        if (ImGui::SliderFloat("Near Clip", &clipN, 0.001f, 10.f))
                            cam->clipNear(clipN);

                        if (ImGui::SliderFloat("Far Clip", &clipF, clipN, std::min(clipF * 1.1f, 1000000.f)))
                            cam->clipFar(clipF);

                        if (ImGui::SliderFloat("Focal Dist.", &focalDist, clipN, clipF))
                            cam->focalDist(focalDist);

                        ImGui::TreePop();
                    }
                }

                // Show special light properties
                if (typeid(*singleNode) == typeid(SLLightSpot) ||
                    typeid(*singleNode) == typeid(SLLightRect) ||
                    typeid(*singleNode) == typeid(SLLightDirect))
                {
                    SLLight* light = nullptr;
                    SLstring typeName;
                    if (typeid(*singleNode) == typeid(SLLightSpot))
                    {
                        light    = (SLLight*)(SLLightSpot*)singleNode;
                        typeName = "Light (spot):";
                    }
                    if (typeid(*singleNode) == typeid(SLLightRect))
                    {
                        light    = (SLLight*)(SLLightRect*)singleNode;
                        typeName = "Light (rectangular):";
                    }
                    if (typeid(*singleNode) == typeid(SLLightDirect))
                    {
                        light    = (SLLight*)(SLLightDirect*)singleNode;
                        typeName = "Light (directional):";
                    }

                    if (light && ImGui::TreeNode(typeName.c_str()))
                    {
                        SLbool on = light->isOn();
                        if (ImGui::Checkbox("Is on", &on))
                            light->isOn(on);

                        ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue;
                        SLCol4f             a     = light->ambientColor();
                        if (ImGui::InputFloat3("Ambient", (float*)&a, 1, flags))
                            light->ambientColor(a);

                        SLCol4f d = light->diffuseColor();
                        if (ImGui::InputFloat3("Diffuse", (float*)&d, 1, flags))
                            light->diffuseColor(d);

                        SLCol4f s = light->specularColor();
                        if (ImGui::InputFloat3("Specular", (float*)&s, 1, flags))
                            light->specularColor(s);

                        float cutoff = light->spotCutOffDEG();
                        if (ImGui::SliderFloat("Spot cut off angle", &cutoff, 0.0f, 180.0f))
                            light->spotCutOffDEG(cutoff);

                        float kc = light->kc();
                        if (ImGui::SliderFloat("Constant attenutation", &kc, 0.0f, 1.0f))
                            light->kc(kc);

                        float kl = light->kl();
                        if (ImGui::SliderFloat("Linear attenutation", &kl, 0.0f, 1.0f))
                            light->kl(kl);

                        float kq = light->kq();
                        if (ImGui::SliderFloat("Quadradic attenutation", &kq, 0.0f, 1.0f))
                            light->kq(kq);

                        ImGui::TreePop();
                    }
                }
            }
            else
            {
                ImGui::Text("No single node selected.");
            }
            ImGui::TreePop();
        }

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
        ImGui::Separator();
        if (ImGui::TreeNode("Single Mesh Properties"))
        {
            if (singleFullMesh)
            {
                SLuint      v = (SLuint)singleFullMesh->P.size();
                SLuint      t = (SLuint)(singleFullMesh->I16.size() ? singleFullMesh->I16.size() : singleFullMesh->I32.size());
                SLMaterial* m = singleFullMesh->mat();
                ImGui::Text("Mesh Name       : %s", singleFullMesh->name().c_str());
                ImGui::Text("No. of Vertices : %u", v);
                ImGui::Text("No. of Triangles: %u", t);

                if (m && ImGui::TreeNode("Material"))
                {
                    ImGui::Text("Material Name: %s", m->name().c_str());

                    if (ImGui::TreeNode("Reflection colors"))
                    {
                        SLCol4f ac = m->ambient();
                        if (ImGui::ColorEdit3("Ambient color", (float*)&ac))
                            m->ambient(ac);

                        SLCol4f dc = m->diffuse();
                        if (ImGui::ColorEdit3("Diffuse color", (float*)&dc))
                            m->diffuse(dc);

                        SLCol4f sc = m->specular();
                        if (ImGui::ColorEdit3("Specular color", (float*)&sc))
                            m->specular(sc);

                        SLCol4f ec = m->emissive();
                        if (ImGui::ColorEdit3("Emissive color", (float*)&ec))
                            m->emissive(ec);

                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNode("Other variables"))
                    {
                        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);

                        SLfloat shine = m->shininess();
                        if (ImGui::SliderFloat("Shininess", &shine, 0.0f, 1000.0f))
                            m->shininess(shine);

                        SLfloat rough = m->roughness();
                        if (ImGui::SliderFloat("Roughness", &rough, 0.0f, 1.0f))
                            m->roughness(rough);

                        SLfloat metal = m->metalness();
                        if (ImGui::SliderFloat("Metalness", &metal, 0.0f, 1.0f))
                            m->metalness(metal);

                        SLfloat kr = m->kr();
                        if (ImGui::SliderFloat("kr", &kr, 0.0f, 1.0f))
                            m->kr(kr);

                        SLfloat kt = m->kt();
                        if (ImGui::SliderFloat("kt", &kt, 0.0f, 1.0f))
                            m->kt(kt);

                        SLfloat kn = m->kn();
                        if (ImGui::SliderFloat("kn", &kn, 1.0f, 2.5f))
                            m->kn(kn);

                        ImGui::PopItemWidth();
                        ImGui::TreePop();
                    }

                    if (m->textures().size() && ImGui::TreeNode("Textures"))
                    {
                        ImGui::Text("No. of textures: %lu", m->textures().size());

                        //SLfloat lineH = ImGui::GetTextLineHeightWithSpacing();
                        SLfloat texW = ImGui::GetWindowWidth() - 4 * ImGui::GetTreeNodeToLabelSpacing() - 10;

                        for (SLuint i = 0; i < m->textures().size(); ++i)
                        {
                            SLGLTexture* t      = m->textures()[i];
                            void*        tid    = (ImTextureID)(intptr_t)t->texID();
                            SLfloat      w      = (SLfloat)t->width();
                            SLfloat      h      = (SLfloat)t->height();
                            SLfloat      h_to_w = h / w;

                            if (ImGui::TreeNode(t->name().c_str()))
                            {
                                ImGui::Text("Size    : %d x %d x %d", t->width(), t->height(), t->depth());
                                ImGui::Text("Type    : %s", t->typeName().c_str());

                                if (t->depth() > 1)
                                {
                                    if (t->target() == GL_TEXTURE_CUBE_MAP)
                                        ImGui::Text("Cube maps can not be displayed.");
                                    else if (t->target() == GL_TEXTURE_3D)
                                        ImGui::Text("3D textures can not be displayed.");
                                }
                                else
                                {
                                    if (typeid(*t) == typeid(SLTransferFunction))
                                    {
                                        SLTransferFunction* tf = (SLTransferFunction*)m->textures()[i];
                                        if (ImGui::TreeNode("Color Points in Transfer Function"))
                                        {
                                            for (SLuint c = 0; c < tf->colors().size(); ++c)
                                            {
                                                SLCol3f color = tf->colors()[c].color;
                                                SLchar  label[20];
                                                sprintf(label, "Color %u", c);
                                                if (ImGui::ColorEdit3(label, (float*)&color))
                                                {
                                                    tf->colors()[c].color = color;
                                                    tf->generateTexture();
                                                }
                                                ImGui::SameLine();
                                                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
                                                sprintf(label, "Pos. %u", c);
                                                SLfloat pos = tf->colors()[c].pos;
                                                if (c > 0 && c < tf->colors().size() - 1)
                                                {
                                                    SLfloat min = tf->colors()[c - 1].pos + 2.0f / (SLfloat)tf->length();
                                                    SLfloat max = tf->colors()[c + 1].pos - 2.0f / (SLfloat)tf->length();
                                                    if (ImGui::SliderFloat(label, &pos, min, max, "%3.2f"))
                                                    {
                                                        tf->colors()[c].pos = pos;
                                                        tf->generateTexture();
                                                    }
                                                }
                                                else
                                                    ImGui::Text("%3.2f Pos. %u", pos, c);
                                                ImGui::PopItemWidth();
                                            }

                                            ImGui::TreePop();
                                        }

                                        if (ImGui::TreeNode("Alpha Points in Transfer Function"))
                                        {
                                            for (SLuint a = 0; a < tf->alphas().size(); ++a)
                                            {
                                                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.25f);
                                                SLfloat alpha = tf->alphas()[a].alpha;
                                                SLchar  label[20];
                                                sprintf(label, "Alpha %u", a);
                                                if (ImGui::SliderFloat(label, &alpha, 0.0f, 1.0f, "%3.2f"))
                                                {
                                                    tf->alphas()[a].alpha = alpha;
                                                    tf->generateTexture();
                                                }
                                                ImGui::SameLine();
                                                sprintf(label, "Pos. %u", a);
                                                SLfloat pos = tf->alphas()[a].pos;
                                                if (a > 0 && a < tf->alphas().size() - 1)
                                                {
                                                    SLfloat min = tf->alphas()[a - 1].pos + 2.0f / (SLfloat)tf->length();
                                                    SLfloat max = tf->alphas()[a + 1].pos - 2.0f / (SLfloat)tf->length();
                                                    if (ImGui::SliderFloat(label, &pos, min, max, "%3.2f"))
                                                    {
                                                        tf->alphas()[a].pos = pos;
                                                        tf->generateTexture();
                                                    }
                                                }
                                                else
                                                    ImGui::Text("%3.2f Pos. %u", pos, a);

                                                ImGui::PopItemWidth();
                                            }

                                            ImGui::TreePop();
                                        }

                                        ImGui::Image(tid, ImVec2(texW, texW * 0.25f), ImVec2(0, 1), ImVec2(1, 0), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));

                                        SLVfloat allAlpha = tf->allAlphas();
                                        ImGui::PlotLines("", allAlpha.data(), (SLint)allAlpha.size(), 0, nullptr, 0.0f, 1.0f, ImVec2(texW, texW * 0.25f));
                                    }
                                    else
                                    {
                                        ImGui::Image(tid, ImVec2(texW, texW * h_to_w), ImVec2(0, 1), ImVec2(1, 0), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));
                                    }
                                }

                                ImGui::TreePop();
                            }
                        }

                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNode("GLSL Program"))
                    {
                        for (SLuint i = 0; i < m->program()->shaders().size(); ++i)
                        {
                            SLGLShader* s     = m->program()->shaders()[i];
                            SLfloat     lineH = ImGui::GetTextLineHeight();

                            if (ImGui::TreeNode(s->name().c_str()))
                            {
                                SLchar text[1024 * 16];
                                strcpy(text, s->code().c_str());
                                ImGui::InputTextMultiline(s->name().c_str(), text, IM_ARRAYSIZE(text), ImVec2(-1.0f, lineH * 16));
                                ImGui::TreePop();
                            }
                        }

                        ImGui::TreePop();
                    }

                    ImGui::TreePop();
                }
            }
            else
            {
                ImGui::Text("No single mesh selected.");
            }

            ImGui::TreePop();
        }

        ImGui::PopStyleColor();
        ImGui::End();
    }
    else if (!singleFullMesh && !s->selectedMeshes().empty())
    {
        // See also SLMesh::handleRectangleSelection
        ImGui::Begin("Properties of Selection", _activator);

        for (auto* selectedNode : s->selectedNodes())
        {
            if (selectedNode->mesh())
            {
                ImGui::Text("Node: %s", selectedNode->name().c_str());
                SLMesh* selectedMesh = selectedNode->mesh();
                if ((SLuint)selectedMesh->IS32.size() > 0)
                {
                    ImGui::Text("   Mesh: %s {%u v.}",
                                selectedMesh->name().c_str(),
                                (SLuint)selectedMesh->IS32.size());
                    ImGui::SameLine();
                    SLstring delBtn = "DEL##" + selectedMesh->name();
                    if (ImGui::Button(delBtn.c_str()))
                    {
                        selectedMesh->deleteSelected(selectedNode);
                    }
                }
            }
        }

        ImGui::End();
    }
    else
    {
        // Nothing is selected
        ImGui::Begin("Properties of Selection", _activator);
        ImGui::Text("There is nothing selected.");
        ImGui::Text("");
        ImGui::Text("Select a single node by");
        ImGui::Text("double-clicking it or");
        ImGui::Text("select multiple nodes by");
        ImGui::Text("SHIFT-double-clicking them.");
        ImGui::Text("");
        ImGui::Text("Select partial meshes by");
        ImGui::Text("CTRL-LMB rectangle drawing.");
        ImGui::Text("");
        ImGui::Text("Press ESC to deselect all.");
        ImGui::Text("");
        ImGui::Text("Be aware that a node may be");
        ImGui::Text("flagged as not selectable.");
        ImGui::End();
    }
    ImGui::PopFont();
}
