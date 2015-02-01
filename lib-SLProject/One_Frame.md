\section oneframe One Frame
This section gives an step by step overview how one frame gets rendered. 

In the most out 

* onFrame:
  * SLScene::updateIfAllViewsGotPainted
     * Calculate the frame time over all views
     * SLAnimManager::update:
     * SLNode::updateAABBRec called on the root node to resize the 
       axis aligned bounding boxes of all nodes that have changed during animation.
     * SLMesh::transformSkin is called on all meshes with skeletons that use SW skinning.
     * SLMesh::updateAccelStruct called on all meshes with software skinning,
  * SLSceneView::onPaint called for every sceneview.
     * SLSceneView::draw3DGL
        * SLCamera::camUpdate updates any camera animation (smooth transitions)
        * All buffers are cleared (color, depth and Occulus frame buffer)
        * Camera Settings:
           * SLCamera::setProjection sets the projection 
             (perspective, orthographic or one of the stereo projections) and viewport.
             On stereo the projection for the left eye is set.
           * SLCamera::setView applies the view transform.
        * Frustum Culling:
           * SLCamera::setFrustumPlanes set the cameras frustum planes according to the
             current projection and view transform.
           * SLNode::cullRec called on the root node:
              * All nodes are checked if they are visible.
                 * All visible nodes without transparencies (opaque) are added to the _opaqueNodes vector.
                 * All visible nodes with transparencies are added to the _blendNodes vector.
           * SLSceneView::draw3DGLAll:
              * Blending is turned off and the depthtest on
              * SLSceneView::draw3DGLNodes is called for every node in the _opaqueNodes vector:
                 * The view matrix is applied to the modelview matrix
                 * The nodes world matrix is applied to the modelview matrix
                 * SLMesh::draw is called on all meshes of the node:
                    * 1) Apply the drawing bits
                    * 2) Apply the uniform variables to the shader
                       * 2a) Activate a shader program if it is not yet in use and apply all its material paramters.
                       * 2b) Pass the modelview and modelview-projection matrix to the shader.
                       * 2c) If needed build and pass the inverse modelview and the normal matrix.
                       * 2d) If the mesh has a skeleton and HW skinning is applied pass the joint matrices.
                    * 3) Build VBOs once
                    * 4) Bind and enable attribute pointers
                    * 5) Finally bind and draw elements
                    * 6) Disable attribute pointers
                    * 7) Draw optional normals & tangents
                    * 8) Draw optional acceleration structure
              * SLSceneView::draw3DGLLines: for every node in the _opaqueNodes vector:
                * The view matrix is applied to the modelview matrix
                * If the drawbit for viewing the AABBs is set SLAABBox::drawWS draws it.
                * If the drawbit for viewing the axis is set SLAABBox::drawAxisWS draws it.
              * SLSceneView::draw3DGLLines: for every node in the _blendNodes vector the same as above.
              * Blending is turned on and the depthtest off.
              * The nodes of the _blendNotes are sorted by depth.
              * SLSceneView::draw3DGLNodes is called for every node in the _blendNodes vector (same as above).
              * Blending is turned off and the depthtest on again.
           * SLMesh::draw3DGLALL is called again for the right eye of stereo projections.
     * SLSceneView::draw2DGL is called for all 2D drawing in orthographic projection.
        * SLSceneView::draw2DGLAll
     * if Oculus stereo projection is used the Oculus frame buffer is drawn and swapped.

