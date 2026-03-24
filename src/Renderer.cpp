#include "../include/Renderer.hpp"


Renderer::Renderer(int width, int height)
    : window_(sf::VideoMode(width, height), "Spring Simulation")
{
    window_.setFramerateLimit(60);
}

bool Renderer::isOpen() const {
    return window_.isOpen();
}



void Renderer::handleEvents(Physics::PhysicString& rope) {
    sf::Event event;

    while (window_.pollEvent(event)) {
        if (event.type == sf::Event::Closed)
            window_.close();

        if (event.type == sf::Event::MouseButtonPressed &&
            event.mouseButton.button == sf::Mouse::Left)
        {
            isDragging_ = true;
        }

        if (event.type == sf::Event::MouseButtonReleased &&
            event.mouseButton.button == sf::Mouse::Left)
        {
            isDragging_ = false;
        }
    }

    // 
    if (isDragging_) {
        sf::Vector2i mousePos = sf::Mouse::getPosition(window_);

        Eigen::Vector2d mouseVec(
            static_cast<double>(mousePos.x),
            static_cast<double>(mousePos.y)
        );

        int tipIndex = rope.getNumPoints() - 1;

        rope.setPointPosition(tipIndex, mouseVec);
        rope.setOldPosition(tipIndex, mouseVec); // Verlet friendly
        rope.setFixed(tipIndex, true);  
    }

    if (event.type == sf::Event::MouseButtonReleased &&
        event.mouseButton.button == sf::Mouse::Left)
    {
        isDragging_ = false;
        rope.setFixed(rope.getNumPoints() - 1, false);  // ← reset
    }
}

void Renderer::drawState(const Eigen::VectorXd& state, int n, sf::Color color)
{
    // state size must be >= 2n
    for (int i = 1; i < n; ++i)
    {
        double x0 = state(i - 1);
        double y0 = state(n + i - 1);
        double x1 = state(i);
        double y1 = state(n + i);

        sf::Vertex line[] = {
            sf::Vertex(sf::Vector2f(x0, y0), color),
            sf::Vertex(sf::Vector2f(x1, y1), color)
        };

        window_.draw(line, 2, sf::Lines);
    }
}


void Renderer::clear() {
    window_.clear(sf::Color::Black);
}

void Renderer::display() {
    window_.display();
}