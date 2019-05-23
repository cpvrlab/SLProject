//#############################################################################
//  File:      SLJob.h
//  Author:    Marcus Hudritsch
//  Date:      December 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLJOB_H
#define SLJOB_H

#include <SL.h>
#include <deque>

//-----------------------------------------------------------------------------
//!
/*!
*/
class SLJob
{
    public:
    SLJob(string name, function<void(SLJob*)> function)
    {
        _name     = name;
        _function = function;
        thread jobThread(function, this);
        jobThread.detach();
        isRunning(true);
    }
    ~SLJob() { ; }

    void update(string progressMsg, int progressNum)
    {
        _mutex.lock();
        _progressMsg = progressMsg;
        _progressNum = progressNum;
        _mutex.unlock();
    }

    void isRunning(bool isRunnung)
    {
        _mutex.lock();
        _isRunning = isRunnung;
        _mutex.unlock();
    }

    private:
    string                 _name;
    string                 _progressMsg;
    int                    _progressNum;
    function<void(SLJob*)> _function;
    bool                   _isRunning;
    mutex                  _mutex;

    public:
    static deque<SLJob*>   queue;
};
//-----------------------------------------------------------------------------
#endif
