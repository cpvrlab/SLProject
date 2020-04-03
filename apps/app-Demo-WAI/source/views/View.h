#ifndef VIEW_H
#define VIEW_H

class View
{
public:
    virtual ~View()
    {
    }

    //! update this view
    virtual bool update() = 0;
};

#endif //VIEW_H
