# Quick Start

<details>
<summary>Installation</summary>

```zsh
pip install -r requirements.txt
```
</details>

<details>
<summary>Demo</summary>

```zsh
python3 main.py
```
or visit `demo.ipynb`
<img alt='result.png' src='assets/result.png' >
</details>

<details>
<summary>Training on custom data</summary>

```zsh
python3 tools/train.py
tensorboard --logdir logs
```

<img alt='assets/acc.png' src='assets/acc.png' width='50%' ><img alt='assets/loss.png' src='assets/loss.png' width='50%' >
</details>

<details>
<summary>Detection from image</summary>

```zsh
python3 tools/detect.py

# 0 9 0 6 0 1 0 2 0
# 8 0 0 0 0 0 0 0 3
# 0 0 3 8 4 2 5 0 0
# 7 0 6 0 0 0 9 0 8
# 0 0 1 0 5 0 7 0 0
# 3 0 5 0 0 0 4 0 6
# 0 0 9 5 1 8 6 0 0
# 4 0 0 0 0 0 0 0 1
# 0 1 0 2 0 4 0 8 0
```
</details>
