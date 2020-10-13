#include <imgui.h>
#include <imgui_internal.h>

#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiSceneGraph.h>

//-----------------------------------------------------------------------------
AppDemoGuiSceneGraph::AppDemoGuiSceneGraph(std::string name, bool* activator, ImFont* font)
  : AppDemoGuiInfosDialog(name, activator, font)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiSceneGraph::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::PushFont(_font);
    ImGui::Begin("Scenegraph", _activator);

    if (s->root3D())
        addSceneGraphNode(s, s->root3D());

    ImGui::End();
    ImGui::PopFont();
}

void AppDemoGuiSceneGraph::addSceneGraphNode(SLScene* s, SLNode* node)
{
    SLbool isSelectedNode = s->singleNodeSelected() == node;
    SLbool isLeafNode = node->children().size() == 0 && node->mesh() == nullptr;

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
        if (node->mesh())
        {
            SLMesh* mesh = node->mesh();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));

            ImGuiTreeNodeFlags meshFlags = ImGuiTreeNodeFlags_Leaf;
            if (s->singleMeshFullSelected() == mesh)
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
