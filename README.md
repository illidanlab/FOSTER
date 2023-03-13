# FOSTER
Code for paper [***Turning the Curse of Heterogeneity in Federated Learning into a Blessing for Out-of-Distribution Detection***](https://openreview.net/pdf?id=mMNimwRb7Gr) by Shuyang Yu, Junyuan Hong, Haotao Wang, Zhangyang Wang and Jiayu Zhou.

## OoD dataset used for evaluation
Links for less common datasets are as follows, [80 Million Tiny Images](http://horatio.cs.nyu.edu/mit/tiny/data/tiny_images.bin)
[Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [Places365](http://places2.csail.mit.edu/download.html), [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz), [LSUN-Resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz), [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz).

## Training & Testing
Example for running foster:
```
#Train
python foster.py --pd_nuser 50 --pu_nclass 3 --wk_iters 5 --model wrn --data stl --con_test_cls --no_track_stat --iter 300 --local_fc --use_external gen_inverse --method crossentropy --score OE --select_generator 1
#Test
python foster.py --pd_nuser 50 --pu_nclass 3 --wk_iters 5 --model wrn --data stl --con_test_cls --no_track_stat --iter 300 --local_fc --use_external gen_inverse --method crossentropy --score OE --select_generator 1 --test --evaluation_score msp
```
Example for running VOS in FL:
```
#Train
python foster.py --pd_nuser 50 --pu_nclass 3 --wk_iters 5 --model wrn --data stl --con_test_cls --no_track_stat --iter 300 --local_fc --use_external None --method crossentropy --score OE --select_generator 1
#Test
python foster.py --pd_nuser 50 --pu_nclass 3 --wk_iters 5 --model wrn --data stl --con_test_cls --no_track_stat --iter 300 --local_fc --use_external None --method crossentropy --score OE --select_generator 1 --test --evaluation_score msp
```
Example for running fedavg:
```
#Train
python foster.py --pd_nuser 50 --pu_nclass 3 --wk_iters 5 --model wrn --data stl --con_test_cls --no_track_stat --iter 300 --local_fc --use_external None --method crossentropy --score OE --select_generator 1 --loss_weight 0
#Test
python foster.py --pd_nuser 50 --pu_nclass 3 --wk_iters 5 --model wrn --data stl --con_test_cls --no_track_stat --iter 300 --local_fc --use_external None --method crossentropy --score OE --select_generator 1 --loss_weight 0 --test --evaluation_score msp
```

Definition for some important parameters:
|Parameter name | Deifinition|
| ------------- |------------|
|pd_nuser|users per domain(for cifar10, cifar100, stl, they only have 1 domain. DomianNet has multiple domains.)|
|pr_nuser|active users per comm round|
|pu_nclass|class per user|
|evaluation_score|post hoc score, eg, msp, Odin, energy, SVM|
