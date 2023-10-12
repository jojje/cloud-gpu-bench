# cloud-gpu-bench
Benchmark script to use for comparing cloud provider GPU performance.

... Or _your own_ graphics card against what the cloud vendors have on offer,
which is the reason this was created in the first place; 
To answer the question: 

> Does it make sense to run _this workload_ in the cloud instead of hogging my
local GPU, and if so, which GPU instance / variant would be at least as fast as
my local one? 

The container runs a GPU [benchmark][1] using PyTorch 2.0.1 and CUDA 11.7.

The image is built automatically from [github][2] and [pushed][3] to dockerhub.

Pre-req:

* To get this to work, the host machine must be able to expose the GPU to the
container. See [nVidia container toolkit][4] for getting that setup, or use a cloud provider
VM image that's already setup with CUDA and docker.

## Usage:

    $ docker run --rm jojje/cloud-gpu-bench --help

    usage: benchmark [OPTIONS]


### Run a short test on all cores for one minute 40 seconds (100 samples)

    $ docker run --rm jojje/cloud-gpu-bench


[1]: https://pypi.org/project/pytorch-benchmark/
[2]: github page
[3]: build log
[4]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
