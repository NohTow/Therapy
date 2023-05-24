
# Therapy

This is the repository for the code of the "Honey, Tell Me What's Wrong", Global Explainability of NLP Models through Cooperative Generation paper.

It introduces a  model-agnostic global explanation method for NLP models that does not require any input data. It means that it provide explanations for any black-box NLP model, not only for an input instance, but for entries on the whole input domain.

It leverages cooperative generation to guide a pre-trained language model with the studied model, to generate text that follow the learned distribution. The distribution of terms in the resulting texts are then used to provide insights on their importance for the model: frequent words are likely to be important for the decision of the model.

To do the cooperative generation, we used the **[PPL-MCTS: Constrained Textual Generation Through Discriminator-Guided Decoding](https://arxiv.org/pdf/2109.13582.pdf)** approach, that leverage Monte Carlo Tree Search to guide the language model using an external model. The original implementation is available [here](https://github.com/NohTow/PPL-MCTS). Therapy simply plug the studied model in the function that take a sequence and return its score to do the generation and then learn a logistic regression model on the produced text. Thus, although we will give some information about the code here, we encourage you to check the original repository for more in-depth details.


## Generation
The first step of Therapy is to cooperatively generate samples for each class of the classifier.
### Parameters
A number of parameters can be defined when executing the MCTS.
|Parameter | Definition |
|--|--|
|\-\-c   |  The exploration constant (c_puct) |
|\-\-temperature  |  Language model temperature when calculating priors|
|\-\-penalty   |  Value of the repetition penalty factor defined in the [CTRL paper](https://arxiv.org/abs/1909.05858)|
|\-\-num_it  |  Number of MCTS iteration for one token|
|\-\-batch_size  |  Batch size|

In our experiments, we set `c` to 5 and the `temperature` to 1.2, in order to encourage the exploration and find most stereotypical words. For the `num_it`, `penalty` and `alpha` parameters, we used the same parameters as in the original paper, respectively 50, 1.2 and 1.
Thus, the command to launch the generation in our setup is:
`python mcts_generation_amazon.py --temperature 1.2 --penalty 1.2 --c 5 --num_it 50 --alpha 1`

## Explanations
Once the samples are generated, a simple logistic regression is learned to classify them using tf-idf representations.
Words that are important for the studied model are likely to be generated frequently and so to have high reglog weights. Using tf-idf allow to filter words that are just frequent across the whole corpus or multiple classes.

To learn the logistic regression on the csv file generated previously, simple run `extract_explanation_agnews.py` by specifying the path of the csv file using the `input` argument, like so:
`python extract_explanation_agnews.py --i agnews_mcts_5.0_1.2_1.2_50_1.csv`

## Reproduction
We provide different elements to enable reproduction and exploration in the `reproduction` folder
### Models
 We share the glass-box models (along with their vectorizer) in the `models` (resp `vectorizer`) folder used for our experiments, to allow the re-generation of samples.
### Generated data
Since generating data with the MCTS can be expensive, we also provide cooperatively generated samples in the `samples` folder.


## License
```
BSD 3-Clause-Attribution License

Copyright (c) 2022, IMATAG and CNRS

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

4. Redistributions of any form whatsoever must retain the following acknowledgment: 
   'This product includes software developed by IMATAG and CNRS'

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
