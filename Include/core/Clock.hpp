#pragma once
#include <SFML/System/Clock.hpp>

namespace Core {

    class Clock {
    private:
        sf::Clock clock_;
        float lastTime_;
        
    public:
        Clock();
        float getDeltaTime();      // time since the last call
        float getElapsedTime();    // total time
        void restart();            // timer restart
    };
}  