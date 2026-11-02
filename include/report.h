// =============================================================================
//  report.h - In bảng kết quả trực quan ra Terminal + ghi CSV cho báo cáo
// =============================================================================
#pragma once
#include "common.h"
#include <string>
#include <vector>

struct Cell {
    std::string text;
    const char* color = "";
    Cell() = default;
    Cell(const std::string& t, const char* c = "") : text(t), color(c) {}
    Cell(const char* t, const char* c = "") : text(t), color(c) {}
};

class Table {
public:
    explicit Table(std::string title) : title_(std::move(title)) {}
    // align: 'l' hoặc 'r' cho từng cột
    void setHeader(const std::vector<std::string>& h, const std::string& align);
    void addRow(const std::vector<Cell>& r);
    void addSeparator();
    void print() const;
private:
    std::string title_;
    std::vector<std::string> header_;
    std::string align_;
    std::vector<std::vector<Cell>> rows_;
    std::vector<bool> sepBefore_;
    bool pendingSep_ = false;
};

class CsvWriter {
public:
    CsvWriter(const std::string& path, const std::vector<std::string>& header);
    ~CsvWriter();
    void row(const std::vector<std::string>& cols);
    bool ok() const { return f_ != nullptr; }
    const std::string& path() const { return path_; }
private:
    FILE* f_ = nullptr;
    std::string path_;
};

// Định dạng số
std::string fmt(const char* f, double v);
std::string fmtMs(double ms);             // 0.123 / 12.35 / 1234.5
std::string fmtCount(u64 v);              // 12,345,678
std::string fmtTeps(double teps);         // 123.4 M / 1.23 G
std::string fmtSpeedup(double s);         // 12.3x
const char* speedupColor(double s);
void printBanner(const std::string& text);
void printSection(const std::string& text);
