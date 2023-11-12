# CUCUMAT -- Cute CUDA Matrix
[CUDA+Python练手项目——Cute CUDA Matrix](https://zhuanlan.zhihu.com/p/663371484)  
This is my private implementation of [cudamat](https://github.com/cudamat/cudamat). Only the matrix operation part of cudamat is implemented. Although the operations implemented have been briefly unit tested, the stability and accuracy of the project as a practice have yet to be tested.
## Build
```bash
mkdir cucumat/build && cd cucumat/build && cmake .. && make && cd -
```
## Example
```python
from cucumat import Matrix
import numpy as np

m, n = np.random.randint(50, 100, (2,))
mat1 = Matrix.matrix_from_array(np.random.randn(m, n))
mat2 = Matrix.matrix_from_array(np.random.randn(m, n))

mat3 = mat1.transpose().dot(mat2)
mat4 = mat2.sum(axis=0)
print(mat4.to_numpy())
```