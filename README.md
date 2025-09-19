

<img src="typhoonMLA-logo.png" alt="logo" width="200"/>

## TyphoonMLA Community Repository

Run TyphoonMLA on your own machine or on a server.

Results are collected from community contributors. **Feel free to create a pull request with your results from different platforms.** We will show all benchmark results in the main page.

## How to run on your local machine

Start a docker container from our pre-built image:
```
docker run -it --rm --gpus=all --runtime=nvidia --name tree-mla acyuzuguler/tree-mla:latest /bin/bash
```

Run experiments and plot results:
```
bash run.sh
```

## How to run it on public cloud infra

Start a VM with the docker image `acyuzuguler/tree-mla:latest`

Run experiments and plot results:
```
bash run.sh
```

## To rebuilt image from scratch

```
docker build -f dockerfiles/Dockerfile -t tree-mla:latest .
```


## Results from community contributors

Add your results below here and create a PR.