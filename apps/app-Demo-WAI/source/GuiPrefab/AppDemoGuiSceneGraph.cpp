#include <imgui.h>
#include <imgui_internal.h>

#include <SLApplication.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiSceneGraph.h>
#include <CVCapture.h>
//-----------------------------------------------------------------------------
AppDemoGuiSceneGraph::AppDemoGuiSceneGraph(std::string name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{ }

//-----------------------------------------------------------------------------
void AppDemoGuiSceneGraph::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Scenegraph", _activator);

    if (s->root3D())
        addSceneGraphNode(s, s->root3D());

    ImGui::End();
}

void AppDemoGuiSceneGraph::addSceneGraphNode(SLScene* s, SLNode* node)
{
    SLbool isSelectedNode = s->selectedNode() == node;
    SLbool isLeafNode     = node->children().size() == 0 && node->meshes().size() == 0;

    ImGuiTreeNodeFlags nodeFlags = 0;
    if (isLeafNode)
        nodeFlags |= ImGuiTreeNodeFlags_Leaf;
    else
        nodeFlags |= ImGuiTreeNodeFlags_OpenOnArrow;

    if (isSelectedNode)
        nodeFlags |= ImGuiTreeNodeFlags_Selected;

    bool nodeIsOpen = ImGui::TreeNodeEx(node->name().c_str(), nodeFlags);

    if (ImGui::IsItemClicked())
        s->selectNodeMesh(node, nullptr);

    if (nodeIsOpen)
    {
        for (auto mesh : node->meshes())
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));

            ImGuiTreeNodeFlags meshFlags = ImGuiTreeNodeFlags_Leaf;
            if (s->selectedMesh() == mesh)
                meshFlags |= ImGuiTreeNodeFlags_Selected;

            ImGui::TreeNodeEx(mesh, meshFlags, "%s", mesh->name().c_str());

            if (ImGui::IsItemClicked())
                s->selectNodeMesh(node, mesh);

            ImGui::TreePop();
            ImGui::PopStyleColor();
        }

        for (auto child : node->children())
            addSceneGraphNode(s, child);

        ImGui::TreePop();
    }
}
