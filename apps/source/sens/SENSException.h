#ifndef SENS_EXCEPTION_H
#define SENS_EXCEPTION_H

#include <exception>
#include <string>
#include <sstream>
#include "SENS.h"

class SENSException : public std::runtime_error
{
public:
	SENSException(SENSType type, const std::string& msg, const int line, const std::string& file)
	 : std::runtime_error(toMessage(msg, line, file).c_str())
	{
	}
	
private:
	std::string toMessage(const std::string& msg, const int line, const std::string& file)
    {
        std::stringstream ss;
        ss << msg << ": Exception thrown at line " << line << " in " << file << std::endl;
        return ss.str();
    }
};

#endif //SENS_EXCEPTION_H