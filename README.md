# Computer Vision Final Project

This project implement a Go Engine.

## How to Wandb
1. pip install wandb
2. wandb login
3. wandb init
4. wandb run python main.py


## Resources

### Reference Paper

1. [Maddison et al. (2015)](http://www.cs.toronto.edu/~cmaddis/pubs/deepgo.pdf)
2. [Tian et al. (2015)](https://arxiv.org/pdf/1511.06410.pdf)
3. AlphaGo
4. AlphaGo Zero

### Publicly Available Go Engines

1. [Pachi](http://pachi.or.cz/), used in Facebook [DeepForest](https://arxiv.org/pdf/1511.06410.pdf) evaluation. Ranking around 2d using single CPU.
2. [GnuGo](https://www.gnu.org/software/gnugo/), used in small model evaluation in Google's [Maddison et al. (2015)](http://www.cs.toronto.edu/~cmaddis/pubs/deepgo.pdf). 

We should implement [**Go Text Protocol**](http://www.lysator.liu.se/~gunnar/gtp/) as this seems to be the common protocol used and there is even a library, [KgsGTP](http://www.gokgs.com/download.jsp) to communicate to KGS Go server.

### Datasets

1. https://github.com/yenw/computer-go-dataset#8-leela-zero-dataset

### SpreadSheet
https://docs.google.com/spreadsheets/d/1-NjOhGRpJHwQ9cG-g5bMRXGC9uE6M0wyL5B4njBk6tU/edit?usp=sharing

### Dependency
### Handle sgf
        pip3 install sgfmill
### Processed Data
https://drive.google.com/drive/folders/1qG6xNuNH-NSjKZMff_PC1SHYkcV0dfeo?usp=sharing
