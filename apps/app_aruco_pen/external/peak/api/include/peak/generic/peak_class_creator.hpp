/*!
 * \file    peak_class_creator.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


namespace
{

/*!
 * \brief Create an instance of a class using their private constructor requiring a backend handle.
 *
 * The constructors, that connect the classes to the C-backend/interface, are private, since they
 * aren't part of the public end-user interface. They are only used from the C++ wrapper classes
 * around the C-backend. So the C++ wrapper classes have access to each other's constructors, you
 * can use this class
 *
 * First, you need "friend" your class to this, so you can call a private constructor:
 *
 * \code
 * private:
 *     friend ClassCreator<Device>;
 *     Device(PEAK_DEVICE_HANDLE deviceHandle, const std::weak_ptr<Interface>& parentInterface);
 * \endcode
 *
 * Then, you can use std::make_shared to create an instance of your class using the private constructor
 *
 * \code
 * std::make_shared<ClassCreator<Device>>(deviceHandle, m_parentInterface);
 * \endcode
 */
template <typename T>
class ClassCreator : public T
{
public:
    template <typename... Args>
    ClassCreator(Args&&... args)
        : T(std::forward<Args>(args)...)
    {}
};

} /* namespace */
