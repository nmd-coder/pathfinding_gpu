// =============================================================================
//  report.cpp - Bảng ASCII căn cột (an toàn với mọi Terminal/code page) + CSV
// =============================================================================
#include "report.h"

#include <algorithm>
#include <cstdarg>
#include <filesystem>

void Table::setHeader(const std::vector<std::string>& h, const std::string& align) {
    header_ = h;
    align_ = align;
    while (align_.size() < header_.size()) align_ += 'r';
}
void Table::addRow(const std::vector<Cell>& r) {
    rows_.push_back(r);
    sepBefore_.push_back(pendingSep_ && !rows_.empty() && rows_.size() > 1);
    pendingSep_ = false;
}
void Table::addSeparator() { pendingSep_ = true; }

void Table::print() const {
    const size_t nc = header_.size();
    std::vector<size_t> w(nc, 0);
    for (size_t i = 0; i < nc; ++i) w[i] = header_[i].size();
    for (auto& r : rows_)
        for (size_t i = 0; i < nc && i < r.size(); ++i) w[i] = std::max(w[i], r[i].text.size());

    auto line = [&](char c) {
        printf("  +");
        for (size_t i = 0; i < nc; ++i) {
            for (size_t k = 0; k < w[i] + 2; ++k) putchar(c);
            putchar('+');
        }
        putchar('\n');
    };
    auto cell = [&](const std::string& t, size_t i, const char* color) {
        size_t pad = w[i] - t.size();
        putchar(' ');
        if (align_[i] == 'r') for (size_t k = 0; k < pad; ++k) putchar(' ');
        if (color && *color) printf("%s%s%s", color, t.c_str(), term::reset());
        else printf("%s", t.c_str());
        if (align_[i] != 'r') for (size_t k = 0; k < pad; ++k) putchar(' ');
        printf(" |");
    };
    if (!title_.empty()) printf("\n  %s%s%s\n", term::bold(), title_.c_str(), term::reset());
    line('-');
    printf("  |");
    for (size_t i = 0; i < nc; ++i) cell(header_[i], i, term::cyan());
    putchar('\n');
    line('=');
    for (size_t r = 0; r < rows_.size(); ++r) {
        if (sepBefore_[r]) line('-');
        printf("  |");
        for (size_t i = 0; i < nc; ++i) {
            const Cell c = i < rows_[r].size() ? rows_[r][i] : Cell("");
            cell(c.text, i, c.color);
        }
        putchar('\n');
    }
    line('-');
    fflush(stdout);
}

CsvWriter::CsvWriter(const std::string& path, const std::vector<std::string>& header) : path_(path) {
    std::error_code ec;
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
    f_ = fopen(path.c_str(), "w");
    if (f_) row(header);
}
CsvWriter::~CsvWriter() { if (f_) fclose(f_); }
void CsvWriter::row(const std::vector<std::string>& cols) {
    if (!f_) return;
    for (size_t i = 0; i < cols.size(); ++i) {
        bool q = cols[i].find(',') != std::string::npos;
        fprintf(f_, "%s%s%s%s", i ? "," : "", q ? "\"" : "", cols[i].c_str(), q ? "\"" : "");
    }
    fputc('\n', f_);
    fflush(f_);
}

std::string fmt(const char* f, double v) {
    char b[64];
    snprintf(b, sizeof(b), f, v);
    return b;
}
std::string fmtMs(double ms) {
    if (ms < 1.0) return fmt("%.3f", ms);
    if (ms < 100.0) return fmt("%.2f", ms);
    return fmt("%.1f", ms);
}
std::string fmtCount(u64 v) {
    std::string s = std::to_string(v), o;
    int c = 0;
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        if (c && c % 3 == 0) o += ',';
        o += *it;
        ++c;
    }
    std::reverse(o.begin(), o.end());
    return o;
}
std::string fmtTeps(double t) {
    if (t >= 1e9) return fmt("%.2f G", t / 1e9);
    if (t >= 1e6) return fmt("%.1f M", t / 1e6);
    return fmt("%.1f K", t / 1e3);
}
std::string fmtSpeedup(double s) { return s >= 100 ? fmt("%.0fx", s) : (s >= 10 ? fmt("%.1fx", s) : fmt("%.2fx", s)); }
const char* speedupColor(double s) { return s >= 1.0 ? term::green() : term::red(); }

void printBanner(const std::string& text) {
    std::string bar(text.size() + 6, '=');
    printf("\n%s%s  %s\n  ## %s ##\n  %s%s\n", term::bold(), term::magenta(), bar.c_str(), text.c_str(), bar.c_str(), term::reset());
}
void printSection(const std::string& text) {
    printf("\n%s%s>>> %s%s\n", term::bold(), term::yellow(), text.c_str(), term::reset());
    fflush(stdout);
}
