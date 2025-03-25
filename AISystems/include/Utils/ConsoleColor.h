#pragma once
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif
#include <io.h> // Äëÿ isatty

class ConsoleColor {
public:
    static void enableColorSupport() {
#ifdef _WIN32
        HANDLE hConsole = GetStdHandle(STD_ERROR_HANDLE);
        if (hConsole == INVALID_HANDLE_VALUE) return;

        DWORD mode;
        if (GetConsoleMode(hConsole, &mode)) {
            mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hConsole, mode);
        }
#endif
    }

    static bool isTerminal() {
#ifdef _WIN32
        return _isatty(_fileno(stderr));
#else
        return isatty(fileno(stderr));
#endif
    }
};