## Train
pc
```
bash train_objpc.sh hanging_mug demo_clean_3d_object_pc 50 0 0 "{A},{B}"

```
baseline
```
bash train.sh hanging_mug demo_clean_3d_object_pc 50 0 0

```
pc+ndf
```
bash train_ndf.sh  hanging_mug demo_clean_3d_object_pc 50 0 0     /home/zheng/train_results/Mug/checkpoints/model_current.pth "" cuda:0 "" "{A},{B}"


```

## Eval
pc
```
bash eval_objpc.sh hanging_mug demo_clean_3d_object_pc 50 0 0 "{A},{B}"

```
baseline
```
bash eval.sh hanging_mug demo_clean_3d_object_pc demo_clean_3d_object_pc 50 0 0

```
pc+ndf
```
bash eval_ndf.sh  hanging_mug demo_clean_3d_object_pc 50 0 0     /home/zheng/train_results/Mug/checkpoints/model_current.pth "" cuda:0 "" "{A},{B}"


```