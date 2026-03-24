#pragma once
#include "../simulation/PhysicSimulator.hpp"
#include <string>
#include <fstream>

namespace Physics {

class Logger {
public:
    Logger(const std::string& node_file, const std::string& edge_file);
    ~Logger();

    void log(const PhysicString& str, int snapshot_id, int frame_id);

private:
    std::ofstream node_out_;
    std::ofstream edge_out_;

    void writeNodeHeader();
    void writeEdgeHeader();
};

} // namespace Physics