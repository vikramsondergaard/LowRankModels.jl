with open('list_landmarks_align_celeba.txt', 'r') as f:
    lines = f.readlines()
    min_dims = 20000
    min_x = 0
    min_y = 0
    for i in range(2, len(lines)):
        l = lines[i].strip('\n').split(' ')
        xs = []
        ys = []
        j = 0
        for e in l:
            if e.isdigit():
                if j % 2 == 0:
                    xs.append(int(e))
                else:
                    ys.append(int(e))
                j += 1
        x_dist = max(xs) - min(xs)
        y_dist = max(ys) - min(ys)
        dims = x_dist * y_dist
        if dims < min_dims:
            min_dims = dims
            min_x = x_dist
            min_y = y_dist
            print(l)
            print(xs)
            print(ys)
    print(f"min_x is {min_x}")
    print(f"min_y is {min_y}")