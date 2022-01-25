//#############################################################################
//  File:      AppPenTrackingEvaluator.h
//  Date:      November 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_APPPENTRACKINGEVALUATOR_H
#define SRC_APPPENTRACKINGEVALUATOR_H

#include <SLVec3.h>
#include <SLVec2.h>
#include <SLEventHandler.h>
#include <SLMaterial.h>
#include <SLSphere.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
//! Evaluates the accuracy of the ArUco pen
/*! When "start" is called, the evaluator places a red sphere on the virtual
 * chessboard that the user has to move the ArUco pen to. When the user presses
 * F7, the offset between the real tip position and the measured tip position
 * is calculated. The dot is then moved to a different location and the process
 * gets repeated. After a certain number of measurements, the results are
 * written to a CSV file in the current working directory.
 */
class AppPenTrackingEvaluator : public SLEventHandler
{
public:
    static AppPenTrackingEvaluator& instance()
    {
        static AppPenTrackingEvaluator instance;
        return instance;
    }

private:
    bool    _isRunning = false;
    int     _x         = 0;
    int     _z         = 0;
    SLNode* _node      = nullptr;

    SLVVec3f                 corners;
    std::vector<SLVec2<int>> intCorners;
    SLVVec3f                 measurements;

public:
    void start();
    bool isRunning() { return _isRunning; }

private:
    void    nextStep();
    void    incCornerPosition();
    SLVec3f currentCorner() const;
    void    finish();

    SLbool onKeyPress(SLKey key,
                      SLKey mod) override;
};
//-----------------------------------------------------------------------------
#endif // SRC_APPPENTRACKINGEVALUATOR_H
