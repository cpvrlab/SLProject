default rel

global aruco_pen_listen
global aruco_pen_publish
global aruco_pen_close
extern WSAStartup
extern WSACleanup
extern getaddrinfo
extern freeaddrinfo
extern socket
extern shutdown
extern closesocket
extern bind
extern connect
extern listen
extern accept
extern recv
extern send
extern malloc
extern free
extern Sleep
extern printf
extern scanf
extern ExitProcess

section .text
aruco_pen_listen:
	sub rsp, 40

		mov r10, rbx
	
	mov qword rcx, string0
	call banjo@print_str
	call socket@ls_new
	mov [main@server], rax
	mov rcx, qword [main@server]
	mov qword rdx, string1
	call socket@ls_listen
	mov qword rcx, string2
	call banjo@print_str
	mov rcx, qword [main@server]
	call socket@ls_accept
	mov [main@client], rax
	mov qword rcx, string3
	call banjo@print_str

		mov rbx, r10
	
	add rsp, 40
	ret
aruco_pen_publish:
	mov [rsp + 8], rcx
	mov [rsp + 16], edx
	sub rsp, 40
ifstart0_0:
	mov rax, qword [main@client]
	mov qword rdx, 0
	cmp rax, rdx
	je ifend0
	mov rcx, qword [main@client]
	mov rdx, qword [rsp + 48]
	mov r8d, dword [rsp + 56]
	call socket@s_send
ifend0:
	add rsp, 40
	ret
aruco_pen_close:
	sub rsp, 40
	mov qword rcx, string4
	call banjo@print_str
ifstart1_0:
	mov rax, qword [main@client]
	mov qword rdx, 0
	cmp rax, rdx
	je ifend1
	mov rcx, qword [main@client]
	call socket@s_delete
ifend1:
ifstart2_0:
	mov rax, qword [main@server]
	mov qword rdx, 0
	cmp rax, rdx
	je ifend2
	mov rcx, qword [main@server]
	call socket@ls_delete
ifend2:
	mov qword rcx, string5
	call banjo@print_str
	add rsp, 40
	ret
socket@MAKEWORD:
	mov [rsp + 8], cl
	mov [rsp + 16], dl
	sub rsp, 40

		movzx ax, cl
		movzx bx, dl
		
		shl ax, 8
		or ax, bx
	
	add rsp, 40
	ret
socket@s_new:
	sub rsp, 40
	call socket@_add_wsa_component
	mov qword rcx, 4
	call memory@alloc
	add rsp, 40
	ret
socket@s_delete:
	mov [rsp + 8], rcx
	sub rsp, 40
	mov rax, [rsp + 48]
	mov ecx, dword [rax + 0]
	mov edx, dword [socket@SD_SEND]
	call shutdown
	mov [rsp + 32], eax
ifstart3_0:
	movsx rax, dword [rsp + 32]
	movsx rdx, dword [socket@SOCKET_ERROR]
	cmp rax, rdx
	jne ifend3
	mov qword rcx, string6
	call banjo@print_str
	mov rax, [rsp + 48]
	mov ecx, dword [rax + 0]
	call closesocket
	call WSACleanup
	mov dword ecx, 1
	call banjo@exit
ifend3:
	mov rax, [rsp + 48]
	mov ecx, dword [rax + 0]
	call closesocket
	call socket@_sub_wsa_component
	add rsp, 40
	ret
socket@s_send:
	mov [rsp + 8], rcx
	mov [rsp + 16], rdx
	mov [rsp + 24], r8d
	sub rsp, 40
	mov rax, [rsp + 48]
	mov ecx, dword [rax + 0]
	mov rdx, qword [rsp + 56]
	mov r8d, dword [rsp + 64]
	mov dword r9d, 0
	call send
	mov [rsp + 32], eax
ifstart4_0:
	movsx rax, dword [rsp + 32]
	movsx rdx, dword [socket@SOCKET_ERROR]
	cmp rax, rdx
	jne ifend4
	mov qword rcx, string7
	call banjo@print_str
	mov rax, [rsp + 48]
	mov ecx, dword [rax + 0]
	call closesocket
	call WSACleanup
	mov dword ecx, 1
	call banjo@exit
ifend4:
	add rsp, 40
	ret
socket@ls_new:
	sub rsp, 40
	call socket@_add_wsa_component
	mov qword rcx, 4
	call memory@alloc
	add rsp, 40
	ret
socket@ls_delete:
	mov [rsp + 8], rcx
	sub rsp, 40
	mov rax, [rsp + 48]
	mov ecx, dword [rax + 0]
	call closesocket
	call socket@_sub_wsa_component
	add rsp, 40
	ret
socket@ls_listen:
	mov [rsp + 8], rcx
	mov [rsp + 16], rdx
	sub rsp, 264
	mov qword [rsp + 32], 0
	mov qword rcx, 48
	call memory@alloc
	mov [rsp + 40], rax
	mov rax, [rsp + 40]
	mov dword [rax + 0], 0
	mov rax, [rsp + 40]
	mov dword [rax + 4], 0
	mov rax, [rsp + 40]
	mov dword [rax + 8], 0
	mov rax, [rsp + 40]
	mov dword [rax + 12], 0
	mov rax, [rsp + 40]
	mov qword [rax + 16], 0
	mov rax, [rsp + 40]
	mov qword [rax + 24], 0
	mov rax, [rsp + 40]
	mov qword [rax + 32], 0
	mov rax, [rsp + 40]
	mov qword [rax + 40], 0
	mov rax, [rsp + 40]
	mov edx, dword [socket@AF_INET]
	mov [rax + 4], edx
	mov rax, [rsp + 40]
	mov edx, dword [socket@SOCK_STREAM]
	mov [rax + 8], edx
	mov rax, [rsp + 40]
	mov edx, dword [socket@IPPROTO_TCP]
	mov [rax + 12], edx
	mov rax, [rsp + 40]
	mov edx, dword [socket@AI_PASSIVE]
	mov [rax + 0], edx
	mov qword rcx, 0
	mov rdx, qword [rsp + 280]
	mov r8, qword [rsp + 40]
	lea rax, [rsp + 32]
	mov r9, rax
	call getaddrinfo
	mov [rsp + 48], eax
ifstart5_0:
	movsx rax, dword [rsp + 48]
	mov qword rdx, 0
	cmp rax, rdx
	je ifend5
	mov qword rcx, string8
	call banjo@print_str
	mov ecx, dword [rsp + 48]
	call banjo@print_int
	call WSACleanup
	mov dword ecx, 1
	call banjo@exit
ifend5:
	mov rax, [rsp + 32]
	mov ecx, dword [rax + 4]
	mov rax, [rsp + 32]
	mov edx, dword [rax + 8]
	mov rax, [rsp + 32]
	mov r8d, dword [rax + 12]
	call socket
	mov [rsp + 52], eax
ifstart6_0:
	movsx rax, dword [rsp + 52]
	movsx rdx, dword [socket@INVALID_SOCKET]
	cmp rax, rdx
	jne ifend6
	mov qword rcx, string9
	call banjo@print_str
	mov rcx, qword [rsp + 32]
	call freeaddrinfo
	call WSACleanup
	mov dword ecx, 1
	call banjo@exit
ifend6:
	mov ecx, dword [rsp + 52]
	mov rax, [rsp + 32]
	mov rdx, qword [rax + 32]
	mov rax, [rsp + 32]
	mov r8d, dword [rax + 16]
	call bind
	mov [rsp + 48], eax
ifstart7_0:
	movsx rax, dword [rsp + 48]
	movsx rdx, dword [socket@SOCKET_ERROR]
	cmp rax, rdx
	jne ifend7
	mov qword rcx, string10
	call banjo@print_str
	mov ecx, dword [rsp + 52]
	call closesocket
	mov rcx, qword [rsp + 32]
	call freeaddrinfo
	call WSACleanup
	mov dword ecx, 1
	call banjo@exit
ifend7:
	mov rcx, qword [rsp + 32]
	call freeaddrinfo
	mov ecx, dword [rsp + 52]
	mov edx, dword [socket@SOMAXCONN]
	call listen
	mov [rsp + 48], eax
ifstart8_0:
	movsx rax, dword [rsp + 48]
	movsx rdx, dword [socket@SOCKET_ERROR]
	cmp rax, rdx
	jne ifend8
	mov qword rcx, string11
	call banjo@print_str
	mov ecx, dword [rsp + 52]
	call closesocket
	call WSACleanup
	mov dword ecx, 1
	call banjo@exit
ifend8:
	mov rax, [rsp + 272]
	mov edx, dword [rsp + 52]
	mov [rax + 0], edx
	add rsp, 264
	ret
socket@ls_accept:
	mov [rsp + 8], rcx
	sub rsp, 72
	mov rax, [rsp + 80]
	mov ecx, dword [rax + 0]
	mov qword rdx, 0
	mov dword r8d, 0
	call accept
	mov [rsp + 32], eax
ifstart9_0:
	movsx rax, dword [rsp + 32]
	movsx rdx, dword [socket@INVALID_SOCKET]
	cmp rax, rdx
	jne ifend9
	mov qword rcx, string12
	call banjo@print_str
	mov rax, [rsp + 80]
	mov ecx, dword [rax + 0]
	call closesocket
	call WSACleanup
	mov dword ecx, 1
	call banjo@exit
ifend9:
	call socket@s_new
	mov [rsp + 36], rax
	mov rax, [rsp + 36]
	mov edx, dword [rsp + 32]
	mov [rax + 0], edx
	mov rax, qword [rsp + 36]
	add rsp, 72
	ret
socket@_add_wsa_component:
	sub rsp, 72
ifstart10_0:
	movsx rax, dword [socket@num_wsa_components]
	mov qword rdx, 0
	cmp rax, rdx
	jne ifend10
	mov qword rcx, 1024
	call memory@alloc
	mov [rsp + 32], rax
	mov byte cl, 2
	mov byte dl, 2
	call socket@MAKEWORD
	mov cx, ax
	mov rdx, qword [rsp + 32]
	call WSAStartup
	mov [rsp + 40], eax
ifstart11_0:
	movsx rax, dword [rsp + 40]
	mov qword rdx, 0
	cmp rax, rdx
	je ifend11
	mov qword rcx, string13
	call banjo@print_str
	mov ecx, dword [rsp + 40]
	call banjo@print_int
	mov dword ecx, 1
	call banjo@exit
ifend11:
ifend10:
	mov eax, dword [socket@num_wsa_components]
	mov dword edx, 1
	add eax, edx
	mov [socket@num_wsa_components], eax
	add rsp, 72
	ret
socket@_sub_wsa_component:
	sub rsp, 40
	mov eax, dword [socket@num_wsa_components]
	mov dword edx, 1
	sub eax, edx
	mov [socket@num_wsa_components], eax
ifstart12_0:
	movsx rax, dword [socket@num_wsa_components]
	mov qword rdx, 0
	cmp rax, rdx
	jne ifend12
	call WSACleanup
ifend12:
	add rsp, 40
	ret
memory@alloc:
	mov [rsp + 8], rcx
	sub rsp, 40
	mov rcx, qword [rsp + 48]
	call malloc
	mov [rsp + 32], rax
	mov rax, qword [rsp + 32]
	add rsp, 40
	ret
banjo@print_str:
	mov [rsp + 8], rcx
	sub rsp, 40
	mov qword rcx, string14
	mov rdx, qword [rsp + 48]
	call printf
	add rsp, 40
	ret
banjo@print_int:
	mov [rsp + 8], ecx
	sub rsp, 40
	mov qword rcx, string15
	movsx rdx, dword [rsp + 48]
	call printf
	add rsp, 40
	ret
banjo@exit:
	mov [rsp + 8], ecx
	sub rsp, 40
	mov ecx, dword [rsp + 48]
	call ExitProcess
	add rsp, 40
	ret

section .data
main@server dq 0
main@client dq 0
string0 db 'starting server...', 0
string1 db '48792', 0
string2 db 'server started, waiting for client to connect', 0
string3 db 'client connected', 0
string4 db 'closing...', 0
string5 db 'server and client closed', 0
socket@AF_UNSPEC dd 0
socket@AF_INET dd 2
socket@SOCK_STREAM dd 1
socket@IPPROTO_TCP dd 6
socket@INVALID_SOCKET dd -1
socket@SOCKET_ERROR dd -1
socket@AI_PASSIVE dd 1
socket@SD_RECEIVE dd 0x00
socket@SD_SEND dd 0x01
socket@SD_BOTH dd 0x02
socket@SOMAXCONN dd 0x7fffffff
socket@num_wsa_components dd 0
string6 db 'shutdown() failed', 0
string7 db 'send() failed', 0
string8 db 'getaddrinfo() failed', 0
string9 db 'socket() failed', 0
string10 db 'bind() failed', 0
string11 db 'listen() failed', 0
string12 db 'accept() failed', 0
string13 db 'WSAStartup() failed!', 0
string14 db '%s', 0xa, '', 0
string15 db '%d', 0xa, '', 0
