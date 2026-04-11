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
bash train_ndf_pointwise.sh hanging_mug demo_clean_3d_object_pc 50 0 0 /home/zheng/train_results/Mug/checkpoints/model_current.pth "" cuda:0 "" "{A},{B}" 128


```
pc+ndf hybrid
```
bash train_ndf_pointwise_hybrid.sh hanging_mug demo_clean_3d_object_pc 50 0 0 /home/zheng/train_results/Mug/checkpoints/model_current.pth "" cuda:0 "" "{A},{B}" 128


```

pc+semantic
```
bash train_semantic_pointwise.sh hanging_mug demo_clean_3d_object_pc 50 0 0 /home/zheng/github/3d_semantic_train/outputs/utonia_universal_field/Mug_semantic/best.pt "" cuda:0 "{A},{B}" 128


```

## Eval
pc
```
bash eval_objpc.sh hanging_mug demo_clean_3d_object_pc demo_clean_3d_object_pc 50 0 0 "{A},{B}"

bash eval_objpc.sh hanging_mug demo_randomized_3d_object_pc demo_clean_3d_object_pc 50 0 0 "{A},{B}"



```
baseline
```
bash eval.sh hanging_mug demo_clean_3d_object_pc demo_clean_3d_object_pc 50 0 0

```
pc+ndf
```
bash eval_ndf_pointwise.sh  hanging_mug demo_clean_3d_object_pc 50 0 0     /home/zheng/train_results/Mug/checkpoints/model_current.pth "" cuda:0 "" "{A},{B}"


```
pc+ndf hybrid
```
bash eval_ndf_pointwise_hybrid.sh  hanging_mug demo_clean_3d_object_pc 50 0 0     /home/zheng/train_results/Mug/checkpoints/model_current.pth "" cuda:0 "" "{A},{B}"


```

pc+ndf
```
bash eval_ndf.sh  hanging_mug demo_clean_3d_object_pc 50 0 0     /home/zheng/train_results/Mug/checkpoints/model_current.pth "" cuda:0 "" "{A},{B}"


```
