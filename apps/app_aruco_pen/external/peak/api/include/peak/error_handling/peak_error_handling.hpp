/*!
 * \file    peak_error_handling.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/exception/peak_exception.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>


namespace peak
{
namespace core
{

inline std::string ReturnCodeToString(PEAK_RETURN_CODE entry)
{
    switch (static_cast<PEAK_RETURN_CODE_t>(entry))
    {
    case PEAK_RETURN_CODE_SUCCESS:
        return "PEAK_RETURN_CODE_SUCCESS";
    case PEAK_RETURN_CODE_ERROR:
        return "PEAK_RETURN_CODE_ERROR";
    case PEAK_RETURN_CODE_NOT_INITIALIZED:
        return "PEAK_RETURN_CODE_NOT_INITIALIZED";
    case PEAK_RETURN_CODE_ABORTED:
        return "PEAK_RETURN_CODE_ABORTED";
    case PEAK_RETURN_CODE_BAD_ACCESS:
        return "PEAK_RETURN_CODE_BAD_ACCESS";
    case PEAK_RETURN_CODE_BAD_ALLOC:
        return "PEAK_RETURN_CODE_BAD_ALLOC";
    case PEAK_RETURN_CODE_BUFFER_TOO_SMALL:
        return "PEAK_RETURN_CODE_BUFFER_TOO_SMALL";
    case PEAK_RETURN_CODE_INVALID_ADDRESS:
        return "PEAK_RETURN_CODE_INVALID_ADDRESS";
    case PEAK_RETURN_CODE_INVALID_ARGUMENT:
        return "PEAK_RETURN_CODE_INVALID_ARGUMENT";
    case PEAK_RETURN_CODE_INVALID_CAST:
        return "PEAK_RETURN_CODE_INVALID_CAST";
    case PEAK_RETURN_CODE_INVALID_HANDLE:
        return "PEAK_RETURN_CODE_INVALID_HANDLE";
    case PEAK_RETURN_CODE_NOT_FOUND:
        return "PEAK_RETURN_CODE_NOT_FOUND";
    case PEAK_RETURN_CODE_OUT_OF_RANGE:
        return "PEAK_RETURN_CODE_OUT_OF_RANGE";
    case PEAK_RETURN_CODE_TIMEOUT:
        return "PEAK_RETURN_CODE_TIMEOUT";
    case PEAK_RETURN_CODE_NOT_AVAILABLE:
        return "PEAK_RETURN_CODE_NOT_AVAILABLE";
    case PEAK_RETURN_CODE_NOT_IMPLEMENTED:
        return "PEAK_RETURN_CODE_NOT_IMPLEMENTED";
    }

    // This shouldn't happen since the switch above covers every value of the switched enum. Nevertheless the
    // following is necessary since it's possible to cast any value in the range of the underlying type to the enum
    // type.
    return "";
}

template <class CallableType>
void ExecuteAndMapReturnCodes(const CallableType& callableObject)
{
    if (callableObject() != PEAK_RETURN_CODE_SUCCESS)
    {
        PEAK_RETURN_CODE lastErrorCode = PEAK_RETURN_CODE_SUCCESS;
        size_t lastErrorDescriptionSize = 0;
        if (PEAK_C_ABI_PREFIX PEAK_Library_GetLastError(&lastErrorCode, nullptr, &lastErrorDescriptionSize)
            != PEAK_RETURN_CODE_SUCCESS)
        {
            throw InternalErrorException("Could not query the last error!");
        }
        std::vector<char> lastErrorDescription(lastErrorDescriptionSize);
        if (PEAK_C_ABI_PREFIX PEAK_Library_GetLastError(
                &lastErrorCode, lastErrorDescription.data(), &lastErrorDescriptionSize)
            != PEAK_RETURN_CODE_SUCCESS)
        {
            throw InternalErrorException("Could not query the last error!");
        }

        std::stringstream errorText;
        errorText << "Error-Code: " << lastErrorCode << " ("
                  << ReturnCodeToString(static_cast<PEAK_RETURN_CODE>(lastErrorCode))
                  << ") | Error-Description: " << lastErrorDescription.data();

        switch (static_cast<PEAK_RETURN_CODE_t>(lastErrorCode))
        {
        case PEAK_RETURN_CODE_SUCCESS:
            return;
        case PEAK_RETURN_CODE_ERROR:
            throw InternalErrorException(errorText.str());
        case PEAK_RETURN_CODE_NOT_INITIALIZED:
            throw NotInitializedException(errorText.str());
        case PEAK_RETURN_CODE_ABORTED:
            throw AbortedException(errorText.str());
        case PEAK_RETURN_CODE_BAD_ACCESS:
            throw BadAccessException(errorText.str());
        case PEAK_RETURN_CODE_BAD_ALLOC:
            throw BadAllocException(errorText.str());
        case PEAK_RETURN_CODE_BUFFER_TOO_SMALL:
            throw InternalErrorException(errorText.str());
        case PEAK_RETURN_CODE_INVALID_ADDRESS:
            throw InvalidAddressException(errorText.str());
        case PEAK_RETURN_CODE_INVALID_ARGUMENT:
            throw InvalidArgumentException(errorText.str());
        case PEAK_RETURN_CODE_INVALID_CAST:
            throw InvalidCastException(errorText.str());
        case PEAK_RETURN_CODE_INVALID_HANDLE:
            throw InvalidInstanceException(errorText.str());
        case PEAK_RETURN_CODE_NOT_FOUND:
            throw NotFoundException(errorText.str());
        case PEAK_RETURN_CODE_OUT_OF_RANGE:
            throw OutOfRangeException(errorText.str());
        case PEAK_RETURN_CODE_TIMEOUT:
            throw TimeoutException(errorText.str());
        case PEAK_RETURN_CODE_NOT_AVAILABLE:
            throw NotAvailableException(errorText.str());
        case PEAK_RETURN_CODE_NOT_IMPLEMENTED:
            throw NotImplementedException(errorText.str());
        }

        // This shouldn't happen since the switch above covers every value of the switched enum. Nevertheless the
        // following is necessary since it's possible to cast any value in the range of the underlying type to the enum
        // type.
        throw InternalErrorException(errorText.str());
    }
}

/* Creates a shared_ptr from weak_ptr and throws an exception if it can't (because the weak_ptr is invalid/expired). */
template <class PointerType>
inline std::shared_ptr<PointerType> LockOrThrow(const std::weak_ptr<PointerType>& pointerToLock)
{
    auto shared = pointerToLock.lock();
    if (!shared)
    {
        throw InternalErrorException("Pointer has expired!");
    }

    return shared;
}

/* Creates a shared_ptr from weak_ptr and throws an exception if it can't (because the weak_ptr is invalid/expired). */
template <class PointerType>
inline std::shared_ptr<PointerType> LockOrThrowOpenedModule(const std::weak_ptr<PointerType>& pointerToLock)
{
    auto shared = pointerToLock.lock();
    if (!shared)
    {
        throw BadAccessException("Associated module is not open!");
    }

    return shared;
}

} /* namespace core */
} /* namespace peak */
