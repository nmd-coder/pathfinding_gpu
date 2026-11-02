#!/usr/bin/env python3
"""Chuyển cú pháp  kernel<<<cfg>>>(args)  thành  pfshim::launch(pfshim::Cfg(cfg), [&]{ kernel(args); })
để mã .cu biên dịch được bằng g++ trong chế độ mô phỏng CPU."""
import sys, re

def match_back_angle(s, j):
    depth = 0
    while j >= 0:
        if s[j] == '>': depth += 1
        elif s[j] == '<':
            depth -= 1
            if depth == 0: return j
        j -= 1
    raise ValueError("unbalanced <>")

def transform(s):
    out, pos = [], 0
    while True:
        i = s.find('<<<', pos)
        if i < 0:
            out.append(s[pos:]); break
        j = i - 1
        while s[j] in ' \t': j -= 1
        if s[j] == '>':
            j = match_back_angle(s, j) - 1
        while j >= 0 and (s[j].isalnum() or s[j] in '_:'): j -= 1
        kstart = j + 1
        kexpr = s[kstart:i].strip()
        k = s.index('>>>', i)
        cfg = s[i + 3:k]
        p = k + 3
        while s[p] in ' \t': p += 1
        assert s[p] == '(', "expected ( after >>>"
        depth, q = 0, p
        while True:
            if s[q] == '(': depth += 1
            elif s[q] == ')':
                depth -= 1
                if depth == 0: break
            q += 1
        args = s[p + 1:q]
        out.append(s[pos:kstart])
        out.append(f"pfshim::launch(pfshim::Cfg({cfg}), [&]() {{ {kexpr}({args}); }})")
        pos = q + 1
    return ''.join(out)

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    open(dst, 'w').write(transform(open(src, encoding='utf-8').read()))
