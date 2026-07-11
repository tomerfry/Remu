/* nolibc acceptance program for remu-user (Linux i386, INT 0x80 syscalls).
 * Prints a greeting, echoes argv (exercising the initial stack image), grows
 * the heap with brk and writes through it, then exits with argc.
 *
 * Rebuild (Linux or WSL; needs only gcc with 32-bit binutils):
 *   gcc -m32 -march=i386 -O1 -nostdlib -static -fno-pie -no-pie \
 *       -o hello-nolibc hello.c
 */

__asm__(
    ".text\n"
    ".globl _start\n"
    "_start:\n"
    "    movl %esp, %eax\n" /* entry ESP: [argc][argv...][envp...] */
    "    andl $-16, %esp\n"
    "    subl $12, %esp\n"
    "    pushl %eax\n"
    "    call cmain\n"
    "    movl %eax, %ebx\n" /* exit status from cmain */
    "    movl $252, %eax\n" /* exit_group */
    "    int  $0x80\n");

static long sys3(long n, long a, long b, long c) {
    long r;
    __asm__ volatile("int $0x80"
                     : "=a"(r)
                     : "a"(n), "b"(a), "c"(b), "d"(c)
                     : "memory");
    return r;
}

static void print(const char *s) {
    long n = 0;
    while (s[n]) n++;
    sys3(4, 1, (long)s, n); /* write(1, s, n) */
}

long cmain(long *sp) {
    long argc = sp[0];
    char **argv = (char **)(sp + 1);

    print("hello via write\n");
    for (long i = 0; i < argc; i++) {
        print("arg: ");
        print(argv[i]);
        print("\n");
    }

    /* Grow the heap with brk and write through the new pages. */
    char *heap = (char *)sys3(45, 0, 0, 0); /* brk(0) */
    char *end = (char *)sys3(45, (long)(heap + 64), 0, 0);
    if (end != heap + 64) {
        print("brk failed\n");
        return 1;
    }
    const char *msg = "heap: brk works\n";
    long i = 0;
    for (; msg[i]; i++) heap[i] = msg[i];
    heap[i] = 0;
    print(heap);

    return argc;
}
