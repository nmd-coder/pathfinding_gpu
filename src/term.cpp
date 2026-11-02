// =============================================================================
//  term.cpp - Màu sắc ANSI cho Terminal (hỗ trợ Windows 10+/Linux)
// =============================================================================
#include "common.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace term {
static bool g_color = false;

void init(bool enableColor) {
    g_color = enableColor;
#ifdef _WIN32
    if (enableColor) {
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode)) {
            if (!SetConsoleMode(h, mode | 0x0004 /*ENABLE_VIRTUAL_TERMINAL_PROCESSING*/)) g_color = false;
        } else {
            g_color = false;   // bị chuyển hướng ra file -> tắt màu
        }
    }
#endif
}
bool colorEnabled() { return g_color; }
const char* reset()   { return g_color ? "\x1b[0m"  : ""; }
const char* bold()    { return g_color ? "\x1b[1m"  : ""; }
const char* green()   { return g_color ? "\x1b[32m" : ""; }
const char* red()     { return g_color ? "\x1b[31m" : ""; }
const char* yellow()  { return g_color ? "\x1b[33m" : ""; }
const char* cyan()    { return g_color ? "\x1b[36m" : ""; }
const char* magenta() { return g_color ? "\x1b[35m" : ""; }
const char* dim()     { return g_color ? "\x1b[2m"  : ""; }
}  // namespace term
