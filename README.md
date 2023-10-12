# cloud-gpu-bench
Benchmark script to use for comparing cloud provider GPU performance.

... Or _your own_ graphics card against what the cloud vendors have on offer,
which is the reason this was created in the first place; 
To answer the question: 

> Does it make sense to run _this workload_ in the cloud instead of hogging my
local GPU, and if so, which GPU instance / variant would be at least as fast as
my local one? 

**Key facts** 

* The container runs the GPU [benchmark][1] using PyTorch 2.0.1 and CUDA 11.7.
* The image is built automatically from [github][2] and [pushed][3] to [dockerhub][4].
* The default test configuration test takes only a few seconds to run.

**Pre-req**

* The host machine must be able to expose the GPU to the container. See
  [nVidia container toolkit][6] for getting that setup, or use a cloud provider
  VM image that's already setup with CUDA and docker.

## Usage:

    $ docker run --rm jojje/cloud-gpu-bench --help

    usage: benchmark [OPTIONS]
    
    Simple pytorch GPU benchmark
    
    options:
      -h, --help      show this help message and exit
      --steps n       Number of training steps (batches) to run (default: 1000)
      --size n        Image size (produces a square dimension image) to train (default: 224)
      --batch-size n  Number of training steps (batches) to run (default: 8)
      --model model   Model to use for the benchmark. Available models: alexnet,
                      convnext_base, convnext_large, convnext_small, convnext_tiny,
                      densenet121, densenet161, densenet169, densenet201, efficientnet_b0,
                      efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4,
                      efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_v2_l,
                      efficientnet_v2_m, efficientnet_v2_s, get_model, get_model_builder,
                      get_model_weights, get_weight, googlenet, inception_v3, list_models,
                      maxvit_t, mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3,
                      mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, regnet_x_16gf,
                      regnet_x_1_6gf, regnet_x_32gf, regnet_x_3_2gf, regnet_x_400mf,
                      regnet_x_800mf, regnet_x_8gf, regnet_y_128gf, regnet_y_16gf,
                      regnet_y_1_6gf, regnet_y_32gf, regnet_y_3_2gf, regnet_y_400mf,
                      regnet_y_800mf, regnet_y_8gf, resnet101, resnet152, resnet18,
                      resnet34, resnet50, resnext101_32x8d, resnext101_64x4d,
                      resnext50_32x4d, shufflenet_v2_x0_5, shufflenet_v2_x1_0,
                      shufflenet_v2_x1_5, shufflenet_v2_x2_0, squeezenet1_0, squeezenet1_1,
                      swin_b, swin_s, swin_t, swin_v2_b, swin_v2_s, swin_v2_t, vgg11,
                      vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, vit_b_16,
                      vit_b_32, vit_h_14, vit_l_16, vit_l_32, wide_resnet101_2,
                      wide_resnet50_2 (default: resnet50)
      --json          Output report in json instead of text form (default: False)

### Example run

    $ docker run --rm --gpus all jojje/cloud-gpu-bench

    [*] initializing benchmark ...
    Warming up with batch_size=1: 100%|█████████████████████████| 1/1 [00:00<00:00,  1.57it/s]
    Warming up with batch_size=1: 100%|████████████████████| 100/100 [00:00<00:00, 282.23it/s]
    Measuring inference for batch_size=1: 100%|██████████| 1000/1000 [00:03<00:00, 263.81it/s]
    Warming up with batch_size=8: 100%|████████████████████| 100/100 [00:00<00:00, 156.09it/s]
    Measuring inference for batch_size=8: 100%|██████████| 1000/1000 [00:06<00:00, 155.10it/s]
    
    ==[ Benchmark result ]=====
    Batches/second  : 155.27 +/- 0.32 [154.15, 155.93]
    MFLOPS          : 3931
    
    ==[ Test configuration ]==
    Model           : resnet50
    Image dimension : (224, 224)
    Batch size      : 8
    Training steps  : 1000
    Model params    : 25557032
    
    ==[ Machine spec ]========
    GPU             : NVIDIA GeForce RTX 3090 (VRAM: 24576.0 MB)
    GPU #           : 1
    CPU             : AMD Ryzen 9 5950X 16-Core Processor, Cores: 16 physical (32 virtual)
    RAM             : 125.51 GB total, 107.99 GB free
    Kernel          : 5.15.0-86-generic

## Tips

If you want to test some heavy model, use large image dimensions or have a
particularly slow GPU you want to test, you can reduce the number of steps to
linearly cut down on the execution time. The default configuration options are
good enough to assess GPUs like-for-like, so I'd recommend you only change them
if you have an explicit purpose for doing so. Reported benchmarks are more
valuable after all, when people use _the same configuration_.


[1]: https://pypi.org/project/pytorch-benchmark/
[2]: https://github.com/jojje/cloud-gpu-bench
[3]: https://github.com/jojje/cloud-gpu-bench/blob/master/.github/workflows/image-pipeline.yaml
[4]: https://hub.docker.com/r/jojje/cloud-gpu-bench/
[5]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
