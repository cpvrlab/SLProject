/*!
 * \file    peak_init_once.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <mutex>


namespace peak
{
namespace core
{

class InitOnce
{
public:
    InitOnce() = default;
    virtual ~InitOnce() = default;
    InitOnce(const InitOnce& other) = delete;
    InitOnce& operator=(const InitOnce& other) = delete;
    InitOnce(InitOnce&& other) = delete;
    InitOnce& operator=(InitOnce&& other) = delete;

protected:
    // override the Initialize() method to lazy-load data in your derived class, then call InitializeIfNecessary() each
    // time before accessing the lazy-load data.
    //
    // Note: If multiple classes in the inheritance-hierarchy use this lazy-loading mechanism, they should call
    //       the Initialize() Method of their parent class, to make sure all levels are initialized.
    virtual void Initialize() const = 0;
    void InitializeIfNecessary() const;

private:
    mutable std::once_flag m_initializedFlag;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline void InitOnce::InitializeIfNecessary() const
{
    std::call_once(m_initializedFlag, [this] { Initialize(); });
}

} /* namespace core */
} /* namespace peak */
