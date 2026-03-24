#include "../Include/app/System.hpp"

int main() {
    App::System system(
        5,        // n
        300.0f,    // k
        5.0f,      // c
        1.0f,      // m
        50.0f,     // L0
        0.02f,     // dt
        0.97f,     // friction
        -13.0f     // kappa
    );

    system.run();
    return 0;
}
