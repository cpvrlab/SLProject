//#############################################################################
//  File:      qtPropertyTreeItem.h
//  Purpose:   Serves as Qt tree widget item representing a property of SLNode
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef QANIMATIONSLIDER_H
#define QANIMATIONSLIDER_H

#include <qevent.h>
#include <qslider.h>
#include <qtooltip.h>
#include <iomanip>

//----------------------------------------------------------------------------
//! Specialized QSlider that provides more mouse interactions and time
class QAnimationSlider : public QSlider
{
public:

    /*! @todo document */
    QAnimationSlider(QWidget* parent) 
        : QSlider(parent) 
    {}
    
    /*! @todo document */
    void setCurrentTime(float time)
    {
        setSliderPosNoSignal(time/_animDuration);
        updateCurrentTimeString();
    }
    
    /*! @todo document */
    void setAnimDuration(float dur) 
    { 
        _animDuration = dur;
        // set slider maximum to millisec of animation length
        setMaximum(floor(_animDuration * 100.0f)); 
    }
    
    /*! @todo document */
    void setSliderPosNoSignal(float normVal) // set slider pos [0,1]
    {
        blockSignals(true);
        setSliderPosition(round(minimum() +normVal * (maximum() - minimum())));
        blockSignals(false);
    }
    
    /*! @todo document */
    float getNormalizedValue() const
    {
        return (float)(value() - minimum()) / (maximum() - minimum());
    }
        
    /*! @todo document */
    void updateCurrentTimeString()
    {
        static int prevPos = value();
        if (prevPos == value())
            return;
        
        _currentTimeString = getTimeStringForPos(value());
    }
    
    /*! @todo document */
    QString getCurrentTimeString() const
    {
        return _currentTimeString;
    }
    
    /*! @todo document */
    QString getDurationTimeString() const
    {
        return getTimeString(_animDuration);
    }
    
    /*! @todo document */
    QString getTimeString(float time) const
    {
        ostringstream oss;
        oss.precision(2);
        oss << setfill('0') << setw(2) << (int)floor(time / 60.0f);
        oss << ":" << fixed << setfill('0') << setw(5) << fmod(time, 60.0f);
        return QString::fromStdString(oss.str());
    }

protected:
    float _currentTime;                 //!< @todo document
    float _animDuration;                //!< @todo document
    QString _currentTimeString;         //!< @todo document
    
    /*! @todo document */
    QString getTimeStringForPos(int sliderPos) const
    {
        float time = (float)sliderPos/100.0f;
        return getTimeString(time);
    }
    
    /*! @todo document */
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
    
    /*! @todo document */
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
    
    /*! @todo document */
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