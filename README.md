## Improving Task-free Continual Learning by Distributionally Robust Memory Evolution (ICML 2022)


## Package Requirements 
- Python 3.8
- Pytorch 1.8.1


**Note :** Our current implementation achieves better results across various hyperparameters than the results in our paper. For example, we can achieve around 38% on CIFAR10 (memory 500), more than 21.5% on CIFAR100 (memory 5000), more than 28% on mini-ImageNet (memory 10000) even combined with simple experience replay (ER) baseline. Our current implementation does not include the gradient dot product constraint since it has little gains but increases computation cost. 


## Download DataSet

Download mini-ImageNet dataset from [here](https://drive.google.com/file/d/1Qkng7kPnL5akXzqvsjLMuP5rFfFZXYy0/view?usp=sharing) and put the dataset into the '/Data' folder.

## Running Experiments


#### Improved Experience Replay

ER baseline + SGLD on CIFAR10:  </br>

`python er_main.py --method SGLD --lr 0.1 --samples_per_task -1 --dataset split_cifar10 --disc_iters 1 --mem_size 50 --suffix 'ER' --buffer_batch_size 10 --batch_size 10 --hyper_search --robust`


ER baseline + SVGD on CIFAR10:  </br>

`python er_main.py --method SVGD --lr 0.1 --samples_per_task -1 --dataset split_cifar10 --disc_iters 1 --mem_size 50 --suffix 'ER' --buffer_batch_size 10 --batch_size 10 --hyper_search --robust`


ER baseline + SGLD on CIFAR100:  </br>

`python er_main.py --method SGLD --lr 0.1 --samples_per_task -1 --dataset split_cifar100 --disc_iters 3 --mem_size 50 --suffix 'ER' --buffer_batch_size 10 --batch_size 10 --hyper_search --robust`


ER baseline + SVGD on CIFAR100:  </br>

`python er_main.py --method SVGD --lr 0.1 --samples_per_task -1 --dataset split_cifar100 --disc_iters 3 --mem_size 50 --suffix 'ER' --buffer_batch_size 10 --batch_size 10 --hyper_search --robust`


ER baseline + SGLD on mini-ImageNet:  </br>

`python er_main.py --method SGLD --lr 0.1 --samples_per_task -1 --dataset miniimagenet --disc_iters 3 --mem_size 100 --suffix 'ER' --buffer_batch_size 10 --batch_size 10 --hyper_search --robust`

ER baseline + SVGD on mini-ImageNet:  </br>

`python er_main.py --method SVGD --lr 0.1 --samples_per_task -1 --dataset miniimagenet --disc_iters 3 --mem_size 100 --suffix 'ER' --buffer_batch_size 10 --batch_size 10 --hyper_search --robust`



## Acknowledgements 
We would like to thank authors of the following repositories </br>
* https://github.com/optimass/Maximally_Interfered_Retrieval



## Cite
```
@inproceedings{wang2022,
  title={Improving Task-free Continual Learning by Distributionally Robust Memory Evolution},
  author={Wang, Zhenyi and Shen, Li and Fang, Le and Suo, Qiuling and Duan, Tiehang and Gao, Mingchen},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```



## Questions?

For general questions, contact [Zhenyi Wang](zhenyiwa@buffalo.edu)  </br>




