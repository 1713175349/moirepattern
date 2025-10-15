# Introduction
生成二维材料双层的moire堆叠转角结构，
要求：两个单层结构的真空层方向在z轴，二维延申方向在xy平面

```
usage: moiregenerate-cmd [-h] [-o OUTPUT] [-r RANGE RANGE RANGE] [-e EPSILON] [-l LEPSILON] [--maxl MAXL] [-m MAXM] [--distance DISTANCE] [--needshift]
                         files files

generate a structure

positional arguments:
  files                 file name

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output dir
  -r RANGE RANGE RANGE, --range RANGE RANGE RANGE
                        theta range: start end num
  -e EPSILON, --epsilon EPSILON
                        epsilon 晶格矢量允许误差
  -l LEPSILON, --lepsilon LEPSILON
                        lepsilon 晶格整体误差
  --maxl MAXL           最大晶格长度
  -m MAXM, --maxm MAXM  maxmium supercell size,搜索的时候建议依次增大，可以避免找不到小原胞
  --distance DISTANCE   distance between two supercells
  --needshift           need shift
```

# Install

`pip install .`


for example:
```
moiregenerate-cmd -o CrI3 -r 0 60 1000 --maxl 50 --distance 3.4 CR1.vasp CR1.vasp
```

