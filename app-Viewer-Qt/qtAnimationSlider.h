//#############################################################################
//  File:      qtPropertyTreeItem.h
//  Purpose:   Serves as Qt tree widget item representing a property of SLNode
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch, Kirchrain 18, 2572 Sutz
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED
//#############################################################################

#ifndef QANIMATIONSLIDER_H
#define QANIMATIONSLIDER_H

#include <qevent.h>
#include <qslider.h>

class QAnimationSlider : public QSlider
{
public:
    QAnimationSlider(QWidget* parent) 
        : QSlider(parent) 
    {}

    void setSliderPosNoSignal(float normVal) // set slider pos [0,1]
    {
        blockSignals(true);
        setSliderPosition(round(minimum() +normVal * (maximum() - minimum())));
        blockSignals(false);
    }

    float getNormalizedValue() const
    {
        return (float)(value() - minimum()) / (maximum() - minimum());
    }

protected:
    void mousePressEvent(QMouseEvent* event)
    {
        if (event->button() == Qt::LeftButton)
        {
            setValueFromRelMousePos(event->x(), event->y());
            event->accept();
        }
        else
            QSlider::mousePressEvent(event);
    }

    void mouseMoveEvent(QMouseEvent* event)
    {
        if (event->buttons() & Qt::LeftButton)
        {
            setValueFromRelMousePos(event->x(), event->y());
            event->accept();
        }
        else
            QSlider::mouseMoveEvent(event);
    }

    void setValueFromRelMousePos(float x, float y)
    {
        // the real clickable slider area isnt as large as (max - min) 
        // we have to consider the slider handles width too
        QStyleOptionSlider opt;
        initStyleOption(&opt);
        QRect handle = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle);
        
        // @todo test vertical implementation (probably inverted)
        if (orientation() == Qt::Vertical)
        {
            float handleH = handle.height();
            float finalY = (y / height()) * (height() + handleH) - (0.5f * handleH);
            setValue(round(minimum() + ((maximum()-minimum()) * finalY / height())));
        }
        else
        {
            float handleW = handle.width();
            float finalX = (x / width()) * (width() + handleW) - (0.5f * handleW);
            setValue(round(minimum() + ((maximum()-minimum()) * finalX / width())));
            //cout << "move to: " << x << " " << handle.width() << " | " << event->x() << " " << event->y() << endl;
        }
    }
};

#endif