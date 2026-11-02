// =============================================================================
//  heuristic.h - Hàm ước lượng khoảng cách h(v) cho A* (dùng chung CPU & GPU)
// =============================================================================
//  h(v) = floor( scale * ||p(v) - p(t)||_2 * (1 - 1e-9) )
//
//  * scale = min_{(u,v) in E} w(u,v) / ||p(u) - p(v)||   (tính lúc dựng đồ thị)
//    => mọi cạnh thỏa  w(u,v) >= scale * ||p(u)-p(v)||.
//  * Theo bất đẳng thức tam giác trong mặt phẳng:
//        scale*||u-t|| <= scale*||u-v|| + scale*||v-t|| <= w(u,v) + scale*||v-t||
//    lấy floor hai vế (w nguyên)  =>  h(u) <= w(u,v) + h(v)   (NHẤT QUÁN / consistent)
//    và h(t) = 0  =>  h là CHẤP NHẬN ĐƯỢC (admissible): h(v) <= d(v,t).
//  * Hệ số (1 - 1e-9) là biên an toàn cho sai số làm tròn dấu phẩy động (FMA trên GPU...).
//  Nhờ vậy A* luôn cho đường đi tối ưu, và có thể kiểm chứng bằng Dijkstra.
// =============================================================================
#pragma once
#include "common.h"
#include <cmath>

PF_HD u32 heuristicValue(double xv, double yv, double xt, double yt, double scale) {
    double dx = xv - xt, dy = yv - yt;
    double e = sqrt(dx * dx + dy * dy) * scale * (1.0 - 1e-9);
    if (!(e > 0.0)) return 0u;
    if (e >= 4.0e9) return 4000000000u;
    return (u32)e;  // floor (e >= 0)
}
