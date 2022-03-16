import os
rows = 1024;
cols = 1024;

for i in range(1) : 
    sparsity = 1e-5
    while (sparsity * rows * cols < 1.2e7 and sparsity < 0.1) :
            os.system('./sparseTest {} {} {} 0 1 1'.format(rows, cols, sparsity)) 
            print('./sparseTest {} {} {} 0 1 1'.format(rows, cols, sparsity)) 
            sparsity = sparsity * 2 
    rows = rows * 2
    cols = cols * 2
