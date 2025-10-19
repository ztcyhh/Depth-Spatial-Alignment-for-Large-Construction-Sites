#!/usr/bin/env python3
import sys
import itertools

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <N_total_lines> <k_train_size>", file=sys.stderr)
        sys.exit(1)

    try:
        N = int(sys.argv[1]) # 总行数
        k = int(sys.argv[2]) # 训练集大小
    except ValueError:
        print(f"Error: N and k must be integers.", file=sys.stderr)
        sys.exit(1)

    if k > N:
        print(f"Error: k ({k}) is larger than total lines N ({N}).", file=sys.stderr)
        sys.exit(1)
    if k < 1 or N < 1:
        print(f"Error: N and k must be positive integers.", file=sys.stderr)
        sys.exit(1)

    # 1. 创建所有行号的列表 (从1开始, 而不是0)
    all_line_nums = set(range(1, N + 1))

    # 2. 生成 C(N, k) 组合
    train_num_combinations = itertools.combinations(all_line_nums, k)

    # 3. 打印 "训练行号;测试行号"
    for train_nums_tuple in train_num_combinations:
        train_nums_set = set(train_nums_tuple)
        
        # 剩下的就是测试集
        test_nums_set = all_line_nums - train_nums_set
        
        # 转换为逗号分隔的字符串
        train_nums_str = ",".join(map(str, train_nums_tuple))
        test_nums_str = ",".join(map(str, sorted(list(test_nums_set))))
        
        # 打印给Bash
        print(f"{train_nums_str};{test_nums_str}")

if __name__ == "__main__":
    main()