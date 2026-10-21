# tinyGpt

This project aims to reimplement openai's GPT2 LLM on a tiny scale, closely follwing the the tutorial by [Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)

tinyGpt is trained on all of Shakespear's works given in input.txt, and the LM is expected to produce Shakespear like output

## First steps

Firstly just a simple bilateral model which doesn't take into account all the tokens that were generated before just the current token was created. The implimentation of which is in `bilateral.py`.

## Self Attention

The implemention can be found in `sdp_attention.py`
Firstly single dot product self attention was applied, that vastly decreased the training loss and validation loss. The model successfully trained at a small scale for lower hyperparemets found at the beginning of the file.

However, as I tried to scale it up, with the current sdp_attention.py my gpu ran out of vram capping at ~ 3.8GB, meanwhile the training needed about 14GB of space.

## Using azure to train models

The plan is to delegate the training to azure cli minimizing cost and maximising speed

TODO:

- add job.yml for azure and try to run in on a T4 machine on the cloud
- ...
