# Computer Vision Final Project

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

### Dependency 
### Handle sgf
        pip3 install sgfmill
