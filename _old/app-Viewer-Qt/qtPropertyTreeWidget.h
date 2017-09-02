//#############################################################################
//  File:      qtPropertyTreeItem.h
//  Purpose:   Serves as Qt tree widget item representing a property of SLNode
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef QTPROPERTYTREEWIDGET_H
#define QTPROPERTYTREEWIDGET_H

#include "../include/SLNode.h"
#include <QTreeWidget>
#include <QTreeWidgetItem>
//----------------------------------------------------------------------------
//! qtPropertyTreeWidget: QTreeWidget derived class for the property treeview.
class qtPropertyTreeWidget : public QTreeWidget
{
    public:
        qtPropertyTreeWidget(QWidget* parent) : QTreeWidget(parent) {;}

        static bool isBeingBuilt;
};
//----------------------------------------------------------------------------
#endif // QTPROPERTYTREEWIDGET_H

