## Improving Task-free Continual Learning by Distributionally Robust Memory Evolution (ICML 2022)


## Package Requirements 
- Python 3.8
- Pytorch 1.8.1




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




