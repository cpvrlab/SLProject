//#############################################################################
//  File:      qtPropertyTreeItem.h
//  Purpose:   Serves as Qt tree widget item representing a property of SLNode
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch, Kirchrain 18, 2572 Sutz
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED
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

