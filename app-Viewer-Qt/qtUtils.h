//#############################################################################
//  File:      qtUtils.h
//  Purpose:   Some static helper functions
//  Author:    Marcus Hudritsch
//  Date:      July 2016
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <QString>
#include <QMessageBox>


#ifndef QTUTILS_H
#define QTUTILS_H

//-----------------------------------------------------------------------------
//! Some Qt helper functions
class qtUtils
{   public:
    static void showMsgBox(QString text)
    {
        QMessageBox msgBox;
        msgBox.setText(text);
        msgBox.exec();
    }
};
//----------------------------------------------------------------------------
#endif