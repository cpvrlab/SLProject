/*!
 * \file    peak_module.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/generic/peak_init_once.hpp>

#include <cstddef>
#include <memory>
#include <vector>


namespace peak
{
namespace core
{

class NodeMap;
class Port;

/*!
 * \brief Represents an extended GenTL port.
 *
 * This class extends the functionality of a GenTL port with the functionality of the GenAPI. Instead of separating the
 * node map from the port, this class brings them together making it much easier to get things done.
 *
 */
class Module : public InitOnce
{
public:
    Module() = default;
    virtual ~Module() = default;
    Module(const Module& other) = delete;
    Module& operator=(const Module& other) = delete;
    Module(Module&& other) = delete;
    Module& operator=(Module&& other) = delete;

    /*!
     * \brief Returns the module's node maps.
     *
     * \return Node maps
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<NodeMap>> NodeMaps() const;
    /*!
     * \brief Returns the module's port.
     *
     * \return Port
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<class Port> Port() const;

protected:
    virtual PEAK_MODULE_HANDLE ModuleHandle() const = 0;

    //! \cond
    void Initialize() const override;
    //! \endcond

private:
    mutable std::vector<std::shared_ptr<NodeMap>> m_nodeMaps;
    mutable std::shared_ptr<class Port> m_port;
};

} /* namespace core */
} /* namespace peak */

#include <peak/node_map/peak_node_map.hpp>
#include <peak/port/peak_port.hpp>


/* Implementation */
namespace peak
{
namespace core
{

inline std::vector<std::shared_ptr<NodeMap>> Module::NodeMaps() const
{
    InitializeIfNecessary();
    return m_nodeMaps;
}

inline std::shared_ptr<class Port> Module::Port() const
{
    InitializeIfNecessary();
    return m_port;
}

//! \cond
inline void Module::Initialize() const
{
    auto moduleHandle = ModuleHandle();

    auto numNodeMaps = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numNodeMaps) {
        return PEAK_C_ABI_PREFIX PEAK_Module_GetNumNodeMaps(moduleHandle, _numNodeMaps);
    });

    std::vector<std::shared_ptr<NodeMap>> nodeMaps;
    for (size_t x = 0; x < numNodeMaps; ++x)
    {
        auto nodeMapHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_MAP_HANDLE>(
            [&](PEAK_NODE_MAP_HANDLE* _nodeMapHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Module_GetNodeMap(moduleHandle, x, _nodeMapHandle);
            });

        nodeMaps.emplace_back(std::make_shared<ClassCreator<NodeMap>>(nodeMapHandle));
    }

    m_nodeMaps = nodeMaps;

    auto portHandle = QueryNumericFromCInterfaceFunction<PEAK_PORT_HANDLE>([&](PEAK_PORT_HANDLE* _portHandle) {
        return PEAK_C_ABI_PREFIX PEAK_Module_GetPort(moduleHandle, _portHandle);
    });

    m_port = std::make_shared<ClassCreator<peak::core::Port>>(portHandle);
}
//! \endcond

} /* namespace core */
} /* namespace peak */
