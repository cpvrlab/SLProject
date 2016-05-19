//#############################################################################
//  File:      ARSceneView.h
//  Purpose:   Augmented Reality Demo
//  Author:    Michael Göttlicher
//  Date:      May 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>

//-----------------------------------------------------------------------------
/*!
SLSceneView derived class for a node transform test application that
demonstrates all transform possibilities in SLNode
*/
class ARSceneView : public SLSceneView
{
    public:
                           ~ARSceneView();
        void initChessboardTracking(string camParamsFilename, int boardHeight, int boardWidth,
            float edgeLengthM );
        void initArucoTracking(string camParamsFilename, int dictionaryId, float markerLength,
            string detectParamFilename );

        // From SLSceneView overwritten
        void                preDraw();
        void                postDraw() {}
        void                postSceneLoad();

    private:
        std::map<int,SLNode*> _arucoNodes;
};
//-----------------------------------------------------------------------------
