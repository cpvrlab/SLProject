/*!
 * \file    peak_exception.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <stdexcept>


namespace peak
{
namespace core
{

/*!
 * \brief The base class for all exceptions thrown by the library.
 */
class Exception : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/*!
 * \brief The exception thrown for signaling an aborted operation.
 */
class AbortedException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for signaling an access error.
 */
class BadAccessException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for signaling a failed memory allocation.
 */
class BadAllocException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for internal errors.
 */
class InternalErrorException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for trying to work on an invalid address.
 */
class InvalidAddressException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for pass an invalid argument to a function.
 */
class InvalidArgumentException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for trying to apply an invalid cast.
 */
class InvalidCastException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for trying to work on an invalid instance.
 */
class InvalidInstanceException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for signaling a feature is not available in the device.
 */
class NotAvailableException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for signaling a failed find operation.
 */
class NotFoundException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for signaling a feature is not implemented.
 */
class NotImplementedException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for signaling that the library was not initialized.
 *
 * \note Remember to call InitializeLibrary() before anything else.
 */
class NotInitializedException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for trying to access a value being out of range.
 */
class OutOfRangeException : public Exception
{
    using Exception::Exception;
};

/*!
 * \brief The exception thrown for signaling an exceeded timeout during a function call.
 */
class TimeoutException : public Exception
{
    using Exception::Exception;
};

} /* namespace core */
} /* namespace peak */
