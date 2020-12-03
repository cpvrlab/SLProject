#include <iostream>
#include <cstring>
#include <resolv.h>
#include <errno.h>
#include <sstream>
#include <netdb.h>
#include <HttpUtils.h>
#include <Utils.h>
using namespace std;

void Socket::reset()
{
    fd    = 0;
    inUse = false;
    memset(&sa, 0, sizeof(sa));
}

int Socket::connectTo(std::string ip, int port)
{
    memset(&sa, 0, sizeof(sa));
    sa.sin_family      = AF_INET;
    sa.sin_addr.s_addr = inet_addr(ip.c_str());
    sa.sin_port        = htons(port);
    addrlen            = sizeof(sa);

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (!fd)
    {
        std::cerr  << "Error creating socket.\n" << std::endl;
        return -1;
    }

    if (connect(fd, (struct sockaddr*)&sa, addrlen))
    {
        std::cerr  << "Error connecting to server.\n" << std::endl;
        return -1;
    }
    inUse = true;
    return 0;
}

int Socket::sendData(const char* data, size_t size)
{
    int len = send(fd, data, size, 0);
    if (len < 0)
        return -1;
    return 0;
}

#define BUFFER_SIZE 1000

std::vector<char> Socket::recieve()
{
    std::vector<char> data;
    data.reserve(BUFFER_SIZE);

    int  len;
    char buf[BUFFER_SIZE];
    do
    {
        len = recv(fd, buf, BUFFER_SIZE, 0);
        if (len < 0)
            return std::vector<char>();

        data.insert(data.end(), buf, buf + len);
    }
    while (len > 0);

    return data;
}

int SecureSocket::connectTo(std::string ip, int port)
{
    Socket::connectTo(ip, port);
    SSL_load_error_strings();
    const SSL_METHOD* meth = TLS_client_method();
    SSL_CTX*          ctx  = SSL_CTX_new(meth);
    ssl                    = SSL_new(ctx);
    if (!ssl)
    {
        std::cerr << "Error creating SSL.\n"
                  << std::endl;
        return -1;
    }
    sslfd = SSL_get_fd(ssl);
    SSL_set_fd(ssl, fd);
    int err = SSL_connect(ssl);
    if (err <= 0)
    {
        std::cerr << "Error creating SSL connection.  err= " << err << std::endl;
        return -1;
    }

    return 0;
}

int SecureSocket::sendData(const char* data, size_t size)
{
    int len = SSL_write(ssl, data, size);
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

std::vector<char> SecureSocket::recieve()
{
    std::vector<char> data;
    data.reserve(BUFFER_SIZE);

    int  len;
    char buf[BUFFER_SIZE];
    do
    {
        len = SSL_read(ssl, buf, BUFFER_SIZE);

        if (len < 0)
            return std::vector<char>();

        data.insert(data.end(), buf, buf + len);
    }
    while (len > 0);

    return data;
}

DNSRequest::DNSRequest(std::string host)
{
    bool hostIsAddr = true;
    for (int c : host)
    {
        if (!isdigit(c) && c != '.') { hostIsAddr = false; }
        break;
    }

    if (hostIsAddr)
    {
        struct sockaddr_in sa;
        sa.sin_family      = AF_INET;
        sa.sin_addr.s_addr = inet_addr(host.c_str());

        struct hostent* h = gethostbyaddr(&sa.sin_addr, sizeof(sa.sin_addr), sa.sin_family);
        if (h != nullptr && h->h_length > 0)
        {
            hostname = std::string(h->h_name);
        }
        addr = host;
    }
    else
    {
        struct hostent* h = gethostbyname(host.c_str());
        if (h != nullptr && h->h_length > 0)
        {
            addr = std::string(inet_ntoa( (struct in_addr) *((struct in_addr *) h->h_addr_list[0])));
        }
        hostname = host;
    }
}

std::string DNSRequest::getAddr()
{
    return addr;
}

std::string DNSRequest::getHostname()
{
    return hostname;
}

HttpUtils::GetRequest::GetRequest(std::string url)
{
    host     = "";
    addr     = "";
    isSecure = false;

    std::string dir = "/";
    std::string tmp;
    size_t      offset = 0;

    if (url.find("https://") != std::string::npos)
    {
        isSecure = true;
        offset   = 8;
    }
    else if (url.find("http://") != std::string::npos)
    {
        offset = 7;
    }

    size_t pos = url.find("/", offset);
    if (pos != std::string::npos)
    {
        dir = url.substr(pos, url.length() - pos);
        if (dir.back() != '/')
            dir.push_back('/');

        tmp = url.substr(offset, pos - offset);
    }
    else
        tmp = url.substr(offset);

    DNSRequest dns(tmp);
    host = dns.getHostname();
    addr = dns.getAddr();

    request = "GET " + dir + " HTTP/1.1\r\nHost: " + host + "\r\n\r\n";
}

int HttpUtils::GetRequest::send()
{
    std::vector<char> data;
    if (isSecure)
    {
        SecureSocket s;
        if (s.connectTo(addr, 443) < 0)
        {
            std::cerr << "could not connect\n" << std::endl;
            return -1;
        }

        s.sendData(request.c_str(), request.length()+1);
        data = s.recieve();
    }
    else
    {
        Socket s;
        if (s.connectTo(addr, 80) < 0)
        {
            std::cerr << "could not connect\n" << std::endl;
            return -1;
        }

        s.sendData(request.c_str(), request.length()+1);
        data = s.recieve();
    }
    std::string h = std::string(data.begin(), data.begin() + 1000);

    size_t contentPos = h.find("\r\n\r\n");
    if (contentPos == std::string::npos)
    {
        std::cerr << "Invalid http response\n" << std::endl;
        return -1;
    }

    headers = std::string(data.begin(), data.begin() + contentPos);

    size_t pos = headers.find("HTTP");
    if (pos != std::string::npos)
    {
        std::string str = headers.substr(pos, headers.find("\r", pos));
        size_t codeIdx;
        for (codeIdx = 0; codeIdx < str.length() && !isdigit(str.at(codeIdx)); codeIdx++);

        if (codeIdx == str.length())
        {
            std::cout << "Invalid http response\n" << std::endl;
            return -1;
        }

        size_t endCodeIdx = str.find(" ", codeIdx);

        statusCode = stoi(str.substr(codeIdx, endCodeIdx));
        status = str.substr(endCodeIdx);
        status = Utils::trimLeftString(Utils::trimRightString(status, " "), " ");
    }

    pos = headers.find("Content-Length:");
    if (pos != std::string::npos)
    {
        std::string str = headers.substr(pos + 15, headers.find("\r", pos));
        contentLength  = stoi(Utils::trimLeftString(Utils::trimRightString(str, " "), " "));
    }

    pos = headers.find("Content-Type:");
    if (pos != std::string::npos)
    {
        std::string str = headers.substr(pos + 13, headers.find("\r", pos));
        contentType = Utils::trimLeftString(Utils::trimRightString(str, " "), " ");
    }

    content = std::vector<char>(data.begin() + contentPos + 4, data.end());

    return 0;
}

std::vector<char> HttpUtils::GetRequest::getContent()
{
    return content;
}

std::vector<std::string> HttpUtils::GetRequest::getListing()
{
    std::string c = std::string(content.data());
    std::vector<string> listing;

    size_t pos = 0;
    size_t end = 0;
    while(1)
    {
        pos = c.find("<", pos);
        end = c.find(">", pos)+1;
        if (pos == std::string::npos || end == std::string::npos)
            break;

        std::string token = c.substr(pos+1, end - pos - 2);
        token = Utils::trimLeftString(Utils::trimRightString(token, " "), " ");

        if (std::string(token.begin(), token.begin()+2) == "a ")
        {
            size_t href = token.find("href");
            if (href != std::string::npos)
            {
                href = token.find("\"", href);
                size_t hrefend = token.find("\"", href + 1);
                if (token.find("?") == std::string::npos)
                {
                    listing.push_back(token.substr(href+1, hrefend - href -1));
                }
            }
        }
        
        pos = end;
    }
    return listing;
}

void HttpUtils::download(std::string url,
                         std::function<void(std::string, std::vector<char>)> f,
                         std::function<void(std::string)> subdir)
{
    GetRequest req = GetRequest(url);
    req.send();

    std::string dir = url;
    Utils::replaceString(dir, "https://", "/");
    Utils::replaceString(dir, "http://", "/");

    if (req.contentType.find("text/html") != std::string::npos)
    {
        if (url.back() != '/')
            url = url + "/";

        std::vector<std::string> listing = req.getListing();

        for (std::string str : listing)
        {   
            if (str.at(0) != '/')
            {
                subdir(dir);
                download(url + str, f, subdir);
            }
        }
    }
    else
    {
        std::vector<char> content = req.getContent();
        f(dir, content);
    }
}

void HttpUtils::download(std::string url, std::string dst)
{
    GetRequest req = GetRequest(url);
    req.send();

    download(url,
    [dst](std::string file, std::vector<char> data)-> void {
        ofstream fout(dst + "/" + file, ios::out | ios::binary);
        fout.write((char*)&data[0], data.size());
        fout.close();
    },
    [dst](std::string dir)-> void {
        Utils::makeDir(dst + dir);
    }
    );
}
