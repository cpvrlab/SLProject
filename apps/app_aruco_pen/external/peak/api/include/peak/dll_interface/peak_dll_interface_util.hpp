/*!
 * \file    peak_dll_interface_util.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/common/peak_common_structs.hpp>
#include <peak/error_handling/peak_error_handling.hpp>
#include <peak/generic/peak_class_creator.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>


namespace
{

inline void CallAndCheckCInterfaceFunction(const std::function<PEAK_RETURN_CODE(void)>& cInterfaceFunction)
{
    peak::core::ExecuteAndMapReturnCodes([&] { return cInterfaceFunction(); });
}

template <class NumericType>
NumericType QueryNumericFromCInterfaceFunction(
    const std::function<PEAK_RETURN_CODE(NumericType* buffer)>& cInterfaceFunction)
{
    NumericType numeric{};

    peak::core::ExecuteAndMapReturnCodes([&] { return cInterfaceFunction(&numeric); });

    return numeric;
}

inline peak::core::RawInformation QueryRawInformationFromCInterfaceFunction(
    const std::function<PEAK_RETURN_CODE(int32_t* dataType, uint8_t* buffer, size_t* bufferSize)>&
        cInterfaceFunction)
{
    int32_t dataType = 0;
    size_t dataSize = 0;

    peak::core::ExecuteAndMapReturnCodes([&] { return cInterfaceFunction(&dataType, nullptr, &dataSize); });
    std::vector<uint8_t> buffer(dataSize);
    peak::core::ExecuteAndMapReturnCodes(
        [&] { return cInterfaceFunction(&dataType, buffer.data(), &dataSize); });

    return peak::core::RawInformation{ dataType, std::move(buffer) };
}

inline std::string QueryStringFromCInterfaceFunction(
    const std::function<PEAK_RETURN_CODE(char* buffer, size_t* bufferSize)>& cInterfaceFunction)
{
    auto strSize = QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* _strSize) { return cInterfaceFunction(nullptr, _strSize); });

    std::vector<char> str(strSize);
    peak::core::ExecuteAndMapReturnCodes([&] { return cInterfaceFunction(str.data(), &strSize); });

    return std::string(str.data(), str.size() - 1);
}

inline std::vector<std::string> QueryStringArrayFromCInterfaceFunction(
    const std::function<PEAK_RETURN_CODE(size_t* arraySize)>& cInterfaceFunctionArraySize,
    const std::function<PEAK_RETURN_CODE(size_t index, char* buffer, size_t* bufferSize)>&
        cInterfaceFunctionArrayMember)
{
    auto arraySize = QueryNumericFromCInterfaceFunction<size_t>(cInterfaceFunctionArraySize);

    std::vector<std::string> array;
    for (size_t x = 0; x < arraySize; ++x)
    {
        array.emplace_back(QueryStringFromCInterfaceFunction(
            [&](char* buffer, size_t* bufferSize) { return cInterfaceFunctionArrayMember(x, buffer, bufferSize); }));
    }

    return array;
}

template <class NumericType>
std::vector<NumericType> QueryNumericArrayFromCInterfaceFunction(
    const std::function<PEAK_RETURN_CODE(NumericType* buffer, size_t* bufferSize)>& cInterfaceFunction)
{
    auto arraySize = QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* _arraySize) { return cInterfaceFunction(nullptr, _arraySize); });

    std::vector<NumericType> array(arraySize);
    peak::core::ExecuteAndMapReturnCodes([&] { return cInterfaceFunction(array.data(), &arraySize); });

    return array;
}

} // namespace
