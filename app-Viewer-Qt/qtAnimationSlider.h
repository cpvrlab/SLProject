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
#include <qtooltip.h>

class QAnimationSlider : public QSlider
{
public:
    QAnimationSlider(QWidget* parent) 
        : QSlider(parent) 
    {}

    void setCurrentTime(float time)
    {
        setSliderPosNoSignal(time/_animDuration);
    }

    void setAnimDuration(float dur) 
    { 
        _animDuration = dur;
        // set slider maximum to millisec of animation length
        setMaximum(floor(_animDuration * 1000)); 
    }

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
        
    QString getCurrentTimeString() const
    {
        return getTimeStringForPos(value());
    }
    QString getDurationTimeString() const
    {
        return getTimeString(_animDuration);
    }

protected:
    float _currentTime;
    float _animDuration;

    QString getTimeStringForPos(int sliderPos) const
    {
        float time = (float)sliderPos/1000.0f;
        return getTimeString(time);
    }

    QString getTimeString(float time) const
    {
        QString timeMin = QString("%1").arg((int)floor(time / 60.0f), 2, 10, QChar('0'));
        QString timeSec = QString("%1").arg((int)floor(fmod(time, 60.0f)), 2, 10, QChar('0'));
        // @todo there is probably an easier way to format the seconds as 00.00 but this has to do for now
        QString timeSecFrac =  QString("%1").arg((int)(SL_fract(time)*100.0f), 2, 10, QChar('0')); 
        return QString(timeMin + ":" + timeSec + "." + timeSecFrac);
    }

    void mousePressEvent(QMouseEvent* event)
    {
        if (event->button() == Qt::LeftButton)
        {
            setValue(mapRelMousePosToValue(event->x(), event->y()));
            event->accept();
        }
        else
            QSlider::mousePressEvent(event);
    }

    void mouseMoveEvent(QMouseEvent* event)
    {
        if (event->buttons() & Qt::LeftButton)
        {
            setValue(mapRelMousePosToValue(event->x(), event->y()));
            event->accept();
        }
        else
            QSlider::mouseMoveEvent(event);

        // show timestamp tooltip
        // @todo    can we somehow get the correct offsets rather than hardcoding the -20 and -50 in
        //          also the tooltip lags a bit behind, check out the tooltip in the vlc player as a goal reference
        QPoint globalPos = mapToGlobal(QPoint(event->x()- 20.0f, y() - 50.0f));
        QToolTip::showText(globalPos,
                           getTimeStringForPos(mapRelMousePosToValue(event->x(), event->y())),
                           this, rect());
    }

    // maps relative mouse coordinates to slider values
    int mapRelMousePosToValue(float x, float y)
    {
        // the real clickable slider area isnt as large as (max - min) 
        // we have to consider the slider handles width too
        int result = 0;
        QStyleOptionSlider opt;
        initStyleOption(&opt);
        QRect handle = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle);

        if (orientation() == Qt::Vertical)
        {
            float handleH = handle.height();
            float finalY = (y / height()) * (height() + handleH) - (0.5f * handleH);
            result = round(minimum() + ((maximum()-minimum()) * finalY / height()));
        }
        else
        {
            float handleW = handle.width();
            float finalX = (x / width()) * (width() + handleW) - (0.5f * handleW);
            result = round(minimum() + ((maximum()-minimum()) * finalX / width()));
            //cout << "move to: " << x << " " << handle.width() << " | " << event->x() << " " << event->y() << endl;
        }

        if (result < minimum())
            result = minimum();
        if (result > maximum())
            result = maximum();

        return result;
    }
};

#endif