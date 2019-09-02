#include <imgui.h>
#include <imgui_internal.h>

#include <SLApplication.h>
#include <AppDemoGuiTransform.h>
#include <AppDemoGuiAbout.h>
//-----------------------------------------------------------------------------
AppDemoGuiTransform::AppDemoGuiTransform(std::string name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiTransform::buildInfos(SLScene * s, SLSceneView * sv)
{
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
    ImGui::Begin("Transform Selected Node", _activator, window_flags);

    if (s->selectedNode())
    {
        SLNode*                 node   = s->selectedNode();
        static SLTransformSpace tSpace = TS_object;
        SLfloat                 t1 = 0.1f, t2 = 1.0f, t3 = 10.0f; // Delta translations
        SLfloat                 r1 = 1.0f, r2 = 5.0f, r3 = 15.0f; // Delta rotations
        SLfloat                 s1 = 1.1f, s2 = 2.0f, s3 = 10.0f; // Scale factors

        // clang-format off
        ImGui::Text("Transf. Space:"); ImGui::SameLine();
        if (ImGui::RadioButton("Object", (int*)&tSpace, 0)) tSpace = TS_object; ImGui::SameLine();
        if (ImGui::RadioButton("World",  (int*)&tSpace, 1)) tSpace = TS_world; ImGui::SameLine();
        if (ImGui::RadioButton("Parent", (int*)&tSpace, 2)) tSpace = TS_parent;
        ImGui::Separator();

        ImGui::Text("Translation X:"); ImGui::SameLine();
        if (ImGui::Button("<<<##Tx")) node->translate(-t3, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button("<<##Tx"))  node->translate(-t2, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button("<##Tx"))   node->translate(-t1, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">##Tx"))   node->translate( t1, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>##Tx"))  node->translate( t2, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>>##Tx")) node->translate( t3, 0, 0, tSpace);

        ImGui::Text("Translation Y:"); ImGui::SameLine();
        if (ImGui::Button("<<<##Ty")) node->translate(0, -t3, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button("<<##Ty"))  node->translate(0, -t2, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button("<##Ty"))   node->translate(0, -t1, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">##Ty"))   node->translate(0,  t1, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>##Ty"))  node->translate(0,  t2, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>>##Ty")) node->translate(0,  t3, 0, tSpace);

        ImGui::Text("Translation Z:"); ImGui::SameLine();
        if (ImGui::Button("<<<##Tz")) node->translate(0, 0, -t3, tSpace); ImGui::SameLine();
        if (ImGui::Button("<<##Tz"))  node->translate(0, 0, -t2, tSpace); ImGui::SameLine();
        if (ImGui::Button("<##Tz"))   node->translate(0, 0, -t1, tSpace); ImGui::SameLine();
        if (ImGui::Button(">##Tz"))   node->translate(0, 0,  t1, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>##Tz"))  node->translate(0, 0,  t2, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>>##Tz")) node->translate(0, 0,  t3, tSpace);

        ImGui::Text("Rotation X   :"); ImGui::SameLine();
        if (ImGui::Button("<<<##Rx")) node->rotate( r3, 1, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button("<<##Rx"))  node->rotate( r2, 1, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button("<##Rx"))   node->rotate( r1, 1, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">##Rx"))   node->rotate(-r1, 1, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>##Rx"))  node->rotate(-r2, 1, 0, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>>##Rx")) node->rotate(-r3, 1, 0, 0, tSpace);

        ImGui::Text("Rotation Y   :"); ImGui::SameLine();
        if (ImGui::Button("<<<##Ry")) node->rotate( r3, 0, 1, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button("<<##Ry"))  node->rotate( r2, 0, 1, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button("<##Ry"))   node->rotate( r1, 0, 1, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">##Ry"))   node->rotate(-r1, 0, 1, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>##Ry"))  node->rotate(-r2, 0, 1, 0, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>>##Ry")) node->rotate(-r3, 0, 1, 0, tSpace);

        ImGui::Text("Rotation Z   :"); ImGui::SameLine();
        if (ImGui::Button("<<<##Rz")) node->rotate( r3, 0, 0, 1, tSpace); ImGui::SameLine();
        if (ImGui::Button("<<##Rz"))  node->rotate( r2, 0, 0, 1, tSpace); ImGui::SameLine();
        if (ImGui::Button("<##Rz"))   node->rotate( r1, 0, 0, 1, tSpace); ImGui::SameLine();
        if (ImGui::Button(">##Rz"))   node->rotate(-r1, 0, 0, 1, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>##Rz"))  node->rotate(-r2, 0, 0, 1, tSpace); ImGui::SameLine();
        if (ImGui::Button(">>>##Rz")) node->rotate(-r3, 0, 0, 1, tSpace);

        ImGui::Text("Scale        :"); ImGui::SameLine();
        if (ImGui::Button("<<<##S"))
            node->scale( s3); ImGui::SameLine();
        if (ImGui::Button("<<##S"))
            node->scale( s2); ImGui::SameLine();
        if (ImGui::Button("<##S"))
            node->scale( s1); ImGui::SameLine();
        if (ImGui::Button(">##S"))
            node->scale(1/s1); ImGui::SameLine();
        if (ImGui::Button(">>##S"))
            node->scale(1/s2); ImGui::SameLine();
        if (ImGui::Button(">>>##S"))
            node->scale(1/s3);
        ImGui::Separator();
        if (ImGui::Button("Reset")) node->om(node->initialOM());

        // clang-format on
    }
    else
    {
        ImGui::Text("No node selected.");
        ImGui::Text("Please select a node by double clicking it.");
    }
    ImGui::End();
    ImGui::PopFont();
}

