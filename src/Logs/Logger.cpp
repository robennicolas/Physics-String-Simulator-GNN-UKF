#include "../../Include/Logs/Logger.hpp"

namespace Physics {

Logger::Logger(const std::string& node_file, const std::string& edge_file) {
    node_out_.open(node_file);
    edge_out_.open(edge_file);

    if (!node_out_.is_open() || !edge_out_.is_open())
        throw std::runtime_error("Logger: could not open output files.");

    writeNodeHeader();
    writeEdgeHeader();
}

Logger::~Logger() {
    if (node_out_.is_open()) node_out_.close();
    if (edge_out_.is_open()) edge_out_.close();
}

void Logger::writeNodeHeader() {
    node_out_ << "snapshot_id,frame_id,node_id,is_fixed,"
              << "x,y,old_x,old_y,"
              << "vx,vy,"
              << "ax,ay\n";
}

void Logger::writeEdgeHeader() {
    edge_out_ << "snapshot_id,frame_id,node_i,node_j,"
              << "length,tension,"
              << "dir_x,dir_y\n";
}

void Logger::log(const PhysicString& str, int snapshot_id, int frame_id) {
    int n = str.getNumPoints();

    // ---- NODES ----
    for (int i = 0; i < n; i++) {
        Eigen::Vector2d pos     = str.getPointPosition(i);   // x, y
        Eigen::Vector2d old_pos = str.getOldPosition(i); // old_x, old_y
        Eigen::Vector2d vel     = str.getPointVelocity(i);
        Eigen::Vector2d acc     = str.getPointAcceleration(i);
        bool fixed              = str.isFixed(i);                  // adapte si besoin

        node_out_ << snapshot_id << ","
                  << frame_id   << ","
                  << i          << ","
                  << fixed << ","
                  << pos.x()    << "," << pos.y()     << ","
                  << old_pos.x()<< "," << old_pos.y() << ","
                  << vel.x()    << "," << vel.y()     << ","
                  << acc.x()    << "," << acc.y()     << "\n";
    }

    // ---- EDGES ----
    for (int i = 0; i < n - 1; i++) {
        edge_out_ << snapshot_id       << ","
                  << frame_id          << ","
                  << i                 << ","
                  << i + 1             << ","
                  << str.getLength(i)   << ","
                  << str.getTension(i)  << ","
                  << str.getDirection(i).x()  << ","
                  << str.getDirection(i).y()  << "\n";
    }
}

} // namespace Physics