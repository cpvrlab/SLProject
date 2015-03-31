//#############################################################################
//  File:      qtNodeTreeItem.h
//  Purpose:   Serves as Qt tree widget item representing an SLNode or SLMesh
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef QTNODETREEITEM_H
#define QTNODETREEITEM_H

#include "../include/SLNode.h"
#include <QTreeWidget>
#include <QTreeWidgetItem>
//----------------------------------------------------------------------------
//! qtNodeTreeItem: QTreeWidgetItem derived class for the scene node treeview.
class qtNodeTreeItem : public QTreeWidgetItem
{
    public:
                        // Constructor for the first item in the tree widget
                        qtNodeTreeItem(SLNode* node,
                                    QTreeWidget* parent) :
                                    QTreeWidgetItem(parent)
                        {  _node = node;
                        _mesh = 0;
                        setText(0,(char*)_node->name().c_str());
                        }
                     
                        // Constructor for an item after a previous one
                        qtNodeTreeItem(SLNode* node,
                                    qtNodeTreeItem* parent) :
                                    QTreeWidgetItem(parent, parent->child(parent->childCount()-1))
                        {  _node = node;
                        _mesh = 0;
                        setText(0,(char*)_node->name().c_str());
                        }

                        // Constructor for an item after a previous one
                        qtNodeTreeItem(SLMesh* mesh,
                                    qtNodeTreeItem* parent) :
                                    QTreeWidgetItem(parent, parent->child(parent->childCount()-1))
                        {   _node = parent->_node;
                        _mesh = mesh;
                        setText(0,(char*)_mesh->name().c_str());
                        setTextColor(0, Qt::darkRed);
                        setBackgroundColor(0, Qt::lightGray);
                        }

        SLNode*        node() {return _node;}
        SLMesh*        mesh() {return _mesh;}

    private:
        SLNode*        _node;
        SLMesh*        _mesh;
};
//----------------------------------------------------------------------------
#endif
