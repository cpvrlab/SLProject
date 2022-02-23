//#############################################################################
//  File:      HttpUtils.cpp
//  Authors:   Luc Girod
//  Date:      2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_BUILD_WITH_OPENSSL
#    include <HttpUtils.h>
#    include <iostream>
#    include <cstring>
#    include <Utils.h>
#    ifdef _WINDOWS
#    else
#        include <netdb.h>
#        include <unistd.h>
#        include <errno.h>
#    endif

//-----------------------------------------------------------------------------
/*!
 *
 */
void Socket::reset()
{
    fd    = 0;
    inUse = false;
    memset(&sa, 0, sizeof(sa));
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param host (ip or hostname)
 * @param port
 * @return
 */
int Socket::connectTo(string ip,
                      int    port)
{
    struct addrinfo* res = NULL;
    int              ret = getaddrinfo(ip.c_str(), NULL, NULL, &res);
    if (ret)
    {
        Utils::log("Socket  ", "invalid address");
        return -1;
    }

    if (res->ai_family == AF_INET)
    {
        ipv = 4;
        memset(&sa, 0, sizeof(sa));
        sa.sin_family = AF_INET;
        sa.sin_addr   = (((struct sockaddr_in*)res->ai_addr))->sin_addr;
        sa.sin_port   = htons(port);
        addrlen       = sizeof(sa);
    }
    else if (res->ai_family == AF_INET6)
    {
        ipv = 6;
        memset(&sa6, 0, sizeof(sa6));
        sa6.sin6_family = AF_INET6;
        sa6.sin6_addr   = (((struct sockaddr_in6*)res->ai_addr))->sin6_addr;
        sa6.sin6_port   = htons(port);
        addrlen         = sizeof(sa6);
    }
    else
    {
        Utils::log("Socket  ", "invalid address");
        return -1;
    }

    fd = (int)socket(res->ai_family, SOCK_STREAM, 0);
    if (!fd)
    {
        Utils::log("Socket  ", "Error creating socket");
        return -1;
    }

    freeaddrinfo(res);

    // Set timeout value to 10s
#    ifndef WINDOWS
    struct timeval tv;
    tv.tv_sec  = 15;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
#    endif

    if (connect(fd, (struct sockaddr*)&sa, addrlen))
    {
        Utils::log("Socket  ", "Error connecting to server.\n");
        return -1;
    }
    inUse = true;
    return 0;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param data
 * @param size
 * @return
 */
int Socket::sendData(const char* data,
                     size_t      size)
{
    int len = (int)send(fd, data, (int)size, 0);
    if (len < 0)
        return -1;
    return 0;
}

//-----------------------------------------------------------------------------
/*!
 *
 * @param data
 * @param size
 * @return
 */
void Socket::disconnect()
{

#    ifdef _WINDOWS
    closesocket(fd);
#    else
    close(fd);
#    endif
}
//-----------------------------------------------------------------------------
/*!
 *
 */
#    define BUFFER_SIZE 500
int Socket::receive(function<int(char* data, int size)> dataCB, int max)
{
    int  n = 0;
    int  len;
    char buf[BUFFER_SIZE];
    do
    {
        if (max != 0 && max - n <= BUFFER_SIZE)
        {
            len = (int)recv(fd, buf, max - n, 0);
            if (len == -1)
            {
#    ifndef _WINDOWS
                if (errno != ECONNRESET)
                    len = (int)recv(fd, buf, max - n, 0);
#    else
                len = (int)recv(fd, buf, max - n, 0);
#    endif

                if (len == -1)
                    return -1;
            }
            if (dataCB(buf, len) != 0)
                return -1;
            return len;
        }
        else
        {
            len = (int)recv(fd, buf, BUFFER_SIZE, 0);

            if (len == -1)
            {
#    ifndef _WINDOWS
                if (errno != ECONNRESET)
                    len = (int)recv(fd, buf, BUFFER_SIZE, 0);
#    else
                len = (int)recv(fd, buf, BUFFER_SIZE, 0);
#    endif
                return -1;
            }

            n = n + len;
            if (dataCB(buf, len) != 0)
                return -1;
        }

    } while (!_interrupt && len > 0);
    _interrupt = false;
    return n;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param ip
 * @param port
 * @return
 */
int SecureSocket::connectTo(string ip, int port)
{
    Socket::connectTo(ip, port);
    SSL_load_error_strings();
    const SSL_METHOD* meth = TLS_client_method();
    SSL_CTX*          ctx  = SSL_CTX_new(meth);
    ssl                    = SSL_new(ctx);
    if (!ssl)
    {
        Utils::log("SecureSocket", "Error creating SSL");
        return -1;
    }
    sslfd = SSL_get_fd(ssl);

    SSL_set_fd(ssl, fd);
    int err = SSL_connect(ssl);
    if (err <= 0)
    {
        Utils::log("SecureSocket", "Error creating SSL connection.  err = %d", err);
        return -1;
    }

    return 0;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param data
 * @param size
 * @return
 */
int SecureSocket::sendData(const char* data, size_t size)
{
    int len = SSL_write(ssl, data, (int)size);
    if (len < 0)
    {
        int err = SSL_get_error(ssl, len);
        switch (err)
        {
            case SSL_ERROR_WANT_WRITE:
                return 0;
            case SSL_ERROR_WANT_READ:
                return 0;
            case SSL_ERROR_ZERO_RETURN:
            case SSL_ERROR_SYSCALL:
            case SSL_ERROR_SSL:
            default:
                return -1;
        }
    }
    return 0;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param dataCB
 * @param max
 */
int SecureSocket::receive(function<int(char* data, int size)> dataCB,
                          int                                 max)
{
    int  len;
    int  n = 0;
    char buf[BUFFER_SIZE];
    do
    {
        if (max != 0 && max - n <= BUFFER_SIZE)
        {
            len = SSL_read(ssl, buf, max - n);
            if (len == -1)
            {
                len = SSL_read(ssl, buf, max - n); // retry
                if (len == -1)
                {
                    Utils::log("SecureSocket", "SSL_read return -1");
                    return -1;
                }
            }

            if (dataCB(buf, len) != 0)
                return -1;
            return len;
        }
        else
        {
            len = SSL_read(ssl, buf, BUFFER_SIZE);
            if (len == -1)
            {
                len = SSL_read(ssl, buf, BUFFER_SIZE); // retry
                if (len == -1)
                {
                    Utils::log("SecureSocket", "SSL_read return -1");
                    return -1;
                }
            }

            n += len;
            if (dataCB(buf, len) != 0)
                return -1;
        }
    } while (!_interrupt && len > 0);
    _interrupt = true;
    return n;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param dataCB
 * @param max
 */
void SecureSocket::disconnect()
{
    SSL_shutdown(ssl);
    SSL_free(ssl);
    ssl = nullptr;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param host
 */
DNSRequest::DNSRequest(string host)
{
    char             s[250];
    int              maxlen = 249;
    struct hostent*  h;
    struct addrinfo* res = NULL;
    int              ret = getaddrinfo(host.c_str(), NULL, NULL, &res);

    if (ret)
    {
        Utils::log("Socket  ", "invalid address");
        hostname = "";
        addr     = "";
        return;
    }

    if (res->ai_family == AF_INET)
    {
        struct sockaddr_in sa;
        memset(&sa, 0, sizeof(sa));
        sa.sin_family = AF_INET;
        sa.sin_addr   = (((struct sockaddr_in*)res->ai_addr))->sin_addr;
        inet_ntop(AF_INET, &(((struct sockaddr_in*)res->ai_addr))->sin_addr, s, maxlen);

#    ifdef _WINDOWS
        h = gethostbyaddr((const char*)&sa.sin_addr, sizeof(sa.sin_addr), sa.sin_family);
#    else
        h = gethostbyaddr(&sa.sin_addr, sizeof(sa.sin_addr), sa.sin_family);
#    endif
    }
    else if (res->ai_family == AF_INET6)
    {
        struct sockaddr_in6 sa;
        memset(&sa, 0, sizeof(sa));
        sa.sin6_family = AF_INET6;
        sa.sin6_addr   = (((struct sockaddr_in6*)res->ai_addr))->sin6_addr;
        inet_ntop(AF_INET6, &(((struct sockaddr_in6*)res->ai_addr))->sin6_addr, s, maxlen);
#    ifdef _WINDOWS
        h = gethostbyaddr((const char*)&sa.sin6_addr, sizeof(sa.sin6_addr), sa.sin6_family);
#    else
        h = gethostbyaddr(&sa.sin6_addr, sizeof(sa.sin6_addr), sa.sin6_family);
#    endif
    }

    if (h != nullptr && h->h_length > 0)
        hostname = string(h->h_name);
    else
        hostname = "";

    addr = string(s);
}
//-----------------------------------------------------------------------------
/*!
 *
 * @return
 */
string DNSRequest::getAddr()
{
    return addr;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @return
 */
string DNSRequest::getHostname()
{
    return hostname;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param data
 * @return
 */
static string base64(const string data)
{
    static constexpr char sEncodingTable[] = {'A',
                                              'B',
                                              'C',
                                              'D',
                                              'E',
                                              'F',
                                              'G',
                                              'H',
                                              'I',
                                              'J',
                                              'K',
                                              'L',
                                              'M',
                                              'N',
                                              'O',
                                              'P',
                                              'Q',
                                              'R',
                                              'S',
                                              'T',
                                              'U',
                                              'V',
                                              'W',
                                              'X',
                                              'Y',
                                              'Z',
                                              'a',
                                              'b',
                                              'c',
                                              'd',
                                              'e',
                                              'f',
                                              'g',
                                              'h',
                                              'i',
                                              'j',
                                              'k',
                                              'l',
                                              'm',
                                              'n',
                                              'o',
                                              'p',
                                              'q',
                                              'r',
                                              's',
                                              't',
                                              'u',
                                              'v',
                                              'w',
                                              'x',
                                              'y',
                                              'z',
                                              '0',
                                              '1',
                                              '2',
                                              '3',
                                              '4',
                                              '5',
                                              '6',
                                              '7',
                                              '8',
                                              '9',
                                              '+',
                                              '/'};

    size_t in_len  = data.size();
    size_t out_len = 4 * ((in_len + 2) / 3);
    string ret(out_len, '\0');
    size_t i;
    char*  p = const_cast<char*>(ret.c_str());

    for (i = 0; i < in_len - 2; i += 3)
    {
        *p++ = sEncodingTable[(data[i] >> 2) & 0x3F];
        *p++ = sEncodingTable[((data[i] & 0x3) << 4) | ((int)(data[i + 1] & 0xF0) >> 4)];
        *p++ = sEncodingTable[((data[i + 1] & 0xF) << 2) | ((int)(data[i + 2] & 0xC0) >> 6)];
        *p++ = sEncodingTable[data[i + 2] & 0x3F];
    }
    if (i < in_len)
    {
        *p++ = sEncodingTable[(data[i] >> 2) & 0x3F];
        if (i == (in_len - 1))
        {
            *p++ = sEncodingTable[((data[i] & 0x3) << 4)];
            *p++ = '=';
        }
        else
        {
            *p++ = sEncodingTable[((data[i] & 0x3) << 4) | ((int)(data[i + 1] & 0xF0) >> 4)];
            *p++ = sEncodingTable[((data[i + 1] & 0xF) << 2)];
        }
        *p++ = '=';
    }

    return ret;
}

//-----------------------------------------------------------------------------
/*!
 *
 * @param url
 * @param host
 * @param path
 * @param useTLS
 */
static void parseURL(string  url,
                     string& host,
                     string& path,
                     bool&   useTLS)
{
    host   = "";
    useTLS = false;

    string dir = "/";
    string tmp;
    size_t offset = 0;

    if (url.find("https://") != string::npos)
    {
        useTLS = true;
        offset = 8;
    }
    else if (url.find("http://") != string::npos)
    {
        offset = 7;
    }

    size_t pos = url.find("/", offset);
    if (pos != string::npos)
    {
        path = url.substr(pos, url.length() - pos);
        host = url.substr(offset, pos - offset);
    }
    else
    {
        path = "/";
        host = url.substr(offset);
    }
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param url
 * @param user
 * @param pwd
 */
HttpUtils::GetRequest::GetRequest(string url,
                                  string user,
                                  string pwd)
{
    string path;
    bool   isSecure;

    parseURL(url, host, path, isSecure);

    DNSRequest dns(host);
    host = dns.getHostname();
    addr = dns.getAddr();

    request = "GET " + path + " HTTP/1.1\r\n";

    if (!user.empty())
    {
        request = request +
                  "Authorization: Basic " +
                  base64(user + ":" + pwd) + "\r\n";
    }
    request = request + "Host: " + host + "\r\n\r\n";

    if (isSecure)
    {
        s    = new SecureSocket();
        port = 443;
    }
    else
    {
        s    = new Socket();
        port = 80;
    }
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param data
 * @return
 */
int HttpUtils::GetRequest::processHttpHeaders(std::vector<char>& data)
{
    string h          = string(data.begin(),
                      data.begin() +
                        (data.size() > 1000 ? 1000 : data.size()));
    size_t contentPos = h.find("\r\n\r\n");
    if (contentPos == string::npos)
    {
        Utils::log("HttpUtils", "Invalid http response");
        return -1;
    }
    headers = string(data.begin(), data.begin() + contentPos);
    contentPos += 4; // to skip "\r\n\r\n" first byte

    std::cout << headers << std::endl;
    size_t pos = headers.find("HTTP");

    if (pos != string::npos)
    {
        string str = headers.substr(pos, headers.find("\r", pos) - pos);
        size_t codeIdx;
        for (codeIdx = 0; codeIdx < str.length() && !isdigit(str.at(codeIdx)); codeIdx++)
            ;

        if (codeIdx == str.length())
        {
            Utils::log("HttpUtils", "Invalid http response");
            return -1;
        }

        size_t endCodeIdx = str.find(" ", codeIdx);

        statusCode = stoi(str.substr(codeIdx, endCodeIdx - codeIdx));
        status     = str.substr(endCodeIdx);
        status     = Utils::trimString(status, " ");
    }

    pos = headers.find("Content-Length:");
    if (pos != string::npos)
    {
        string str    = headers.substr(pos + 16,
                                    headers.find("\r", pos + 16) - pos - 16);
        contentLength = stoi(Utils::trimString(str, " "));
    }

    pos = headers.find("Content-Type:");
    if (pos != string::npos)
    {
        string str  = headers.substr(pos + 13,
                                    headers.find("\r", pos + 13) - pos - 13);
        contentType = Utils::trimString(str, " ");
    }
    return (int)contentPos;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @return
 */
int HttpUtils::GetRequest::send()
{
    if (s->connectTo(addr, port) < 0)
    {
        Utils::log("HttpUtils", "Could not connect");
        return -1;
    }

    s->sendData(request.c_str(), request.length() + 1);

    std::vector<char>* v = &firstBytes;
    int                ret;
    ret = s->receive([v](char* buf, int size) -> int
                     {
        v->reserve(v->size() + size);
        copy(&buf[0], &buf[size], back_inserter(*v));
        return 0; },
                     1000);

    contentOffset = processHttpHeaders(firstBytes);
    s->disconnect();
    if (ret == -1)
        return 1;
    return 0;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param contentCB
 */
int HttpUtils::GetRequest::getContent(function<int(char* buf, int size)> contentCB)
{
    if (s->connectTo(addr, port) < 0)
    {
        Utils::log("HttpUtils", "Could not connect");
        return 1;
    }

    s->sendData(request.c_str(), request.length() + 1);
    std::vector<char>* v = &firstBytes;

    s->receive([v](char* buf, int size) -> int
               {
        v->reserve(v->size() + size);
        copy(&buf[0], &buf[size], back_inserter(*v));
        return 0; },
               contentOffset);

    int ret = s->receive(contentCB, 0);
    s->disconnect();
    return ret;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @return
 */
std::vector<string> HttpUtils::GetRequest::getListing()
{
    std::vector<char> content;
    getContent([&content](char* buf, int size) -> int
               {
        content.reserve(content.size() + size);
        copy(&buf[0], &buf[size], back_inserter(content));
        return 0; });

    string              c = string(content.data());
    std::vector<string> listing;

    size_t pos = 0;
    size_t end = 0;
    while (1)
    {
        pos = c.find("<", pos);
        end = c.find(">", pos) + 1;
        if (pos == string::npos || end == string::npos)
            break;

        string token = c.substr(pos + 1, end - pos - 2);
        token        = Utils::trimString(token, " ");

        if (string(token.begin(), token.begin() + 2) == "a ")
        {
            size_t href = token.find("href");
            if (href != string::npos)
            {
                href           = token.find("\"", href);
                size_t hrefend = token.find("\"", href + 1);
                if (token.find("?") == string::npos)
                {
                    token = token.substr(href + 1, hrefend - href - 1);
                    if (token == "../" || token == ".")
                    {
                        pos = end;
                        continue;
                    }
                    listing.push_back(token);
                }
            }
        }
        pos = end;
    }
    return listing;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param url
 * @param processFile
 * @param writeChunk
 * @param processDir
 * @param user
 * @param pwd
 * @param base
 */
int HttpUtils::download(string                                               url,
                        function<int(string path, string file, size_t size)> processFile,
                        function<int(char* data, int size)>                  writeChunk,
                        function<int(string)>                                processDir,
                        string                                               user,
                        string                                               pwd,
                        string                                               base)
{
    HttpUtils::GetRequest req = HttpUtils::GetRequest(url, user, pwd);
    if (req.send() < 0)
        return SERVER_NOT_REACHABLE;
    base = Utils::unifySlashes(base);

    if (req.contentType == "text/html")
    {
        if (url.back() != '/')
            url = url + "/";

        if (!processDir(base))
            return CANT_CREATE_DIR;

        std::vector<string> listing = req.getListing();

        for (string str : listing)
        {
            if (str.at(0) != '/')
                return download(url + str,
                                processFile,
                                writeChunk,
                                processDir,
                                user,
                                pwd,
                                base + str);
        }
    }
    else
    {
        string file = url.substr(url.rfind("/") + 1);

        int possibleSplit = (int)base.rfind(file);
        if (possibleSplit != string::npos && base.size() - possibleSplit - 1 == file.size())
            base = base.substr(0, possibleSplit);

        base = Utils::unifySlashes(base);

        if (processFile(base, file, req.contentLength) != 0)
            return CANT_CREATE_FILE;

        if (req.getContent(writeChunk) == -1)
            return CONNECTION_CLOSED;

        return 0;
    }
    return 1;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param url
 * @param dst
 * @param user
 * @param pwd
 * @param progress
 */
int HttpUtils::download(string                                      url,
                        string                                      dst,
                        string                                      user,
                        string                                      pwd,
                        function<int(size_t curr, size_t filesize)> progress)
{
    std::ofstream fs;
    size_t        totalBytes  = 0;
    size_t        writtenByte = 0;

    bool dstIsDir = true;

    if (!Utils::fileExists(dst))
    {
        Utils::makeDirRecurse(dst);
        dstIsDir = true;
    }

    return download(
      url,
      [&fs, &totalBytes, &dst, &dstIsDir](string path,
                                          string file,
                                          size_t size) -> int
      {
          try
          {
              if (dstIsDir)
                  fs.open(path + file, std::ios::out | std::ios::binary);
              else
                  fs.open(dst, std::ios::out | std::ios::binary);
          }
          catch (std::exception& e)
          {
              std::cerr << e.what() << '\n';
              return 1;
          }
          totalBytes = size;
          return 0;
      },
      [&fs, progress, &writtenByte, &totalBytes](char* data, int size) -> int
      {
          if (size > 0)
          {
              try
              {
                  fs.write(data, size);
                  if (progress && progress(writtenByte += size, totalBytes) != 0)
                  {
                      fs.close();
                      return 1;
                  }
              }
              catch (const std::exception& e)
              {
                  std::cerr << e.what() << '\n';
                  return 1;
              }
              return 0;
          }
          else
          {
              if (progress)
                  progress(totalBytes, totalBytes);
              fs.close();
              return 0;
          }
      },
      [&dstIsDir](string dir) -> int
      {
          if (!dstIsDir)
              return 1;
          return Utils::makeDir(dir) == 0;
      },
      user,
      pwd,
      dst);
}
//-----------------------------------------------------------------------------
/*!
 * HttpUtils::download return 0 on success otherwise an error code if the
 * download of the file at url to the destination at dst was successful.
 * @param url A uniform resource locator file address of the file to download
 * @param dst A uniform resource locator folder address as destination folder
 * @param progress A function object that is called during the progress
 */
int HttpUtils::download(string                                      url,
                        string                                      dst,
                        function<int(size_t curr, size_t filesize)> progress)
{
    return download(url, dst, "", "", progress);
}
//-----------------------------------------------------------------------------

int HttpUtils::length(string url, string user, string pwd)
{
    HttpUtils::GetRequest req = HttpUtils::GetRequest(url, user, pwd);
    if (req.send() < 0)
        return -1;

    return (int)req.contentLength;
}
//-----------------------------------------------------------------------------

#endif // SL_BUILD_WITH_OPENSSL
