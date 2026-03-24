#pragma once

#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include "Simulation/PhysicSimulator.hpp"

class Renderer {
public:
    Renderer(int width, int height);

    bool isOpen() const;
    void handleEvents(Physics::PhysicString& rope);


    void drawState(const Eigen::VectorXd& state, int n, sf::Color color);

    void clear();
    void display();


private:
    sf::RenderWindow window_;
    bool isDragging_ = false;
};
